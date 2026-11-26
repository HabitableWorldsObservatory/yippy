"""JAX functions for image processing operations."""

import jax.numpy as jnp
from hwoutils.fft import (
    fft_shear_setup as fft_shear_setup,
)
from hwoutils.fft import (
    fft_shear_x as fft_shear_x,
)
from hwoutils.fft import (
    fft_shear_y as fft_shear_y,
)
from hwoutils.fft import (
    fft_shift_x,
    fft_shift_y,
)
from hwoutils.precision import like
from jax import lax


def create_shift_mask(psf, shift_x, shift_y, x_grid, y_grid, fill_val=1):
    """Create a mask to identify valid pixels to average.

    This function is useful because when the PSF is shifted there are empty
    pixels, since they were outside the initial image, and should not be
    included in the final average. This version avoids lax.cond for better
    performance under JIT.

    Args:
        psf:
            The PSF image to shift, shape (height, width). Only its shape and
            dtype are used.
        shift_x:
            The shift in the x direction, in pixels.
        shift_y:
            The shift in the y direction, in pixels.
        x_grid:
            The x pixel-index grid, broadcastable against the PSF shape.
        y_grid:
            The y pixel-index grid, broadcastable against the PSF shape.
        fill_val:
            The value of the valid pixels in the mask. Defaults to 1.

    Returns:
        The mask, shape of the broadcast grids, with the PSF dtype. Valid
        pixels hold ``fill_val`` and pixels shifted in from outside the image
        hold zero.
    """
    height, width = psf.shape

    # Create masks for x and y shifts using branchless arithmetic.
    # This is much more efficient than using lax.cond.

    # For positive x shift, valid pixels are where x_grid >= ceil(shift_x).
    # For negative x shift, valid pixels are where x_grid < width - ceil(-shift_x).
    shift_x_pos = jnp.maximum(0, shift_x)
    shift_x_neg = jnp.maximum(0, -shift_x)
    n_x_pos = jnp.ceil(shift_x_pos).astype(jnp.int32)
    n_x_neg = jnp.ceil(shift_x_neg).astype(jnp.int32)
    mask_x = (x_grid >= n_x_pos) & (x_grid < (width - n_x_neg))

    # For positive y shift, valid pixels are where y_grid >= ceil(shift_y).
    # For negative y shift, valid pixels are where y_grid < height - ceil(-shift_y).
    shift_y_pos = jnp.maximum(0, shift_y)
    shift_y_neg = jnp.maximum(0, -shift_y)
    n_y_pos = jnp.ceil(shift_y_pos).astype(jnp.int32)
    n_y_neg = jnp.ceil(shift_y_neg).astype(jnp.int32)
    mask_y = (y_grid >= n_y_pos) & (y_grid < (height - n_y_neg))

    # Combine the masks and multiply by the fill value
    mask = mask_x & mask_y
    return mask.astype(psf.dtype) * fill_val


def get_near_inds_offsets_1D(
    x_offsets: jnp.ndarray, y_offsets: jnp.ndarray, _x: float, _y: float
):
    """Computes the nearest indices and offsets for the given _x position in 1D.

    Args:
        x_offsets: 1D array of x offsets, sorted in ascending order.
        y_offsets: 1D array of y offsets. Only the first entry is used.
        _x: The x-coordinate for which to find nearby offsets, in the same
            unit as ``x_offsets``.
        _y: The y-coordinate. Unused in 1D.

    Returns:
        A tuple ``(near_inds, near_offsets)``. ``near_inds`` has shape (2, 2)
        and holds the (x index, y index) pairs of the two surrounding points.
        ``near_offsets`` has shape (2, 2) and holds the corresponding
        (x, y) offsets.
    """
    # Find insertion index for _x
    _x_ind = jnp.searchsorted(x_offsets, _x, side="left")

    # Handle boundary conditions
    x_ind_low = jnp.clip(_x_ind - 1, 0, x_offsets.size - 1)
    x_ind_high = jnp.clip(_x_ind, 0, x_offsets.size - 1)

    y_ind = 0

    # Collect the two nearest indices
    near_inds = jnp.array([[x_ind_low, y_ind], [x_ind_high, y_ind]])

    # Extract the corresponding x offset values
    x_vals_low = x_offsets[x_ind_low]
    x_vals_high = x_offsets[x_ind_high]
    y_val = y_offsets[y_ind]

    near_offsets = jnp.array([[x_vals_low, y_val], [x_vals_high, y_val]])

    return near_inds, near_offsets


def create_avg_psf_1D(
    x: float,
    y: float,
    pixel_scale_arcsec: float,
    x_offsets: jnp.ndarray,
    y_offsets: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    reshaped_psfs: jnp.ndarray,
):
    """Creates and returns the PSF at the specified off-axis position using JAX.

    Args:
        x: X position of the source, in the same unit as ``x_offsets``.
        y: Y position of the source, in the same unit as ``y_offsets``.
        pixel_scale_arcsec: Pixel scale, in the unit of the offsets per pixel.
        x_offsets: Sorted 1D array of tabulated x offsets.
        y_offsets: Sorted 1D array of tabulated y offsets.
        x_grid: The x pixel-index grid used to mask the shifted PSFs.
        y_grid: The y pixel-index grid used to mask the shifted PSFs.
        reshaped_psfs: Tabulated PSFs, shape (n_x_offsets, n_y_offsets, height,
            width).

    Returns:
        The PSF at the requested position, shape (height, width): the
        Gaussian-weighted average of the shifted neighboring PSFs, normalized
        by the weight mask.
    """
    # The core logic is similar to OffAx, but uses JAX operations
    _x, _y = convert_xy_1D(x, y)

    near_inds, near_offsets = get_near_inds_offsets_1D(x_offsets, y_offsets, _x, _y)

    near_psfs = reshaped_psfs[near_inds[:, 0], near_inds[:, 1]]

    near_shifts = (jnp.array([_x, _y]) - near_offsets) / pixel_scale_arcsec
    near_diffs = jnp.linalg.norm(near_shifts, axis=1)
    sigma = 0.25
    # Adding a small value to avoid division by zero
    weights = jnp.exp(-(near_diffs**2) / (2 * sigma**2)) + 1e-16
    weights /= weights.sum()

    # Manually shift each PSF
    shifted_psf1, mask1 = shift_and_mask(
        near_psfs[0],
        near_shifts[0, 0],
        near_shifts[0, 1],
        weights[0],
        x_grid,
        y_grid,
    )
    shifted_psf2, mask2 = shift_and_mask(
        near_psfs[1],
        near_shifts[1, 0],
        near_shifts[1, 1],
        weights[1],
        x_grid,
        y_grid,
    )

    # Accumulate the weighted PSFs and the weight masks
    psf = shifted_psf1 + shifted_psf2
    weight_array = mask1 + mask2

    # Normalize the PSF
    safe_reciprocal = jnp.where(weight_array != 0, 1.0 / weight_array, 0.0)
    psf = psf * safe_reciprocal

    return psf


def create_avg_psf_2DQ(
    x: float,
    y: float,
    pixel_scale_arcsec: float,
    x_offsets: jnp.ndarray,
    y_offsets: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    reshaped_psfs: jnp.ndarray,
):
    """Creates and returns the PSF at the specified off-axis position using JAX.

    Args:
        x: X position of the source, in the same unit as ``x_offsets``.
        y: Y position of the source, in the same unit as ``y_offsets``.
        pixel_scale_arcsec: Pixel scale, in the unit of the offsets per pixel.
        x_offsets: Sorted 1D array of tabulated x offsets.
        y_offsets: Sorted 1D array of tabulated y offsets.
        x_grid: The x pixel-index grid used to mask the shifted PSFs.
        y_grid: The y pixel-index grid used to mask the shifted PSFs.
        reshaped_psfs: Tabulated PSFs, shape (n_x_offsets, n_y_offsets, height,
            width).

    Returns:
        The PSF at the requested position, shape (height, width): the
        Gaussian-weighted average of the shifted neighboring PSFs, normalized
        by the weight mask.
    """
    # The core logic is similar to OffAx, but uses JAX operations
    _x, _y = convert_xy_2DQ(x, y)

    near_inds, near_offsets = get_near_inds_offsets_2D(x_offsets, y_offsets, _x, _y)

    near_psfs = reshaped_psfs[near_inds[:, 0], near_inds[:, 1]]

    near_shifts = (jnp.array([_x, _y]) - near_offsets) / pixel_scale_arcsec
    near_diffs = jnp.linalg.norm(near_shifts, axis=1)
    sigma = 0.25
    # Adding a small value to avoid division by zero
    weights = jnp.exp(-(near_diffs**2) / (2 * sigma**2)) + 1e-16
    weights /= weights.sum()

    # Manually shift each PSF
    shifted_psf1, mask1 = shift_and_mask(
        near_psfs[0],
        near_shifts[0, 0],
        near_shifts[0, 1],
        weights[0],
        x_grid,
        y_grid,
    )
    shifted_psf2, mask2 = shift_and_mask(
        near_psfs[1],
        near_shifts[1, 0],
        near_shifts[1, 1],
        weights[1],
        x_grid,
        y_grid,
    )
    shifted_psf3, mask3 = shift_and_mask(
        near_psfs[2],
        near_shifts[2, 0],
        near_shifts[2, 1],
        weights[2],
        x_grid,
        y_grid,
    )
    shifted_psf4, mask4 = shift_and_mask(
        near_psfs[3],
        near_shifts[3, 0],
        near_shifts[3, 1],
        weights[3],
        x_grid,
        y_grid,
    )

    # Accumulate the weighted PSFs and the weight masks
    psf = shifted_psf1 + shifted_psf2 + shifted_psf3 + shifted_psf4
    weight_array = mask1 + mask2 + mask3 + mask4

    safe_reciprocal = jnp.where(weight_array != 0, 1.0 / weight_array, 0.0)
    psf = psf * safe_reciprocal

    return psf


def get_near_inds_offsets_2D(
    x_offsets: jnp.ndarray, y_offsets: jnp.ndarray, _x: float, _y: float
):
    """Computes the nearest indices and offsets for the given (_x, _y) position in 2D.

    Args:
        x_offsets: 1D array of x offsets, sorted in ascending order.
        y_offsets: 1D array of y offsets, sorted in ascending order.
        _x: The x-coordinate for which to find nearby offsets, in the same
            unit as ``x_offsets``.
        _y: The y-coordinate for which to find nearby offsets, in the same
            unit as ``y_offsets``.

    Returns:
        A tuple ``(near_inds, near_offsets)``. ``near_inds`` has shape (4, 2)
        and holds the (x index, y index) pairs of the four surrounding points.
        ``near_offsets`` has shape (4, 2) and holds the corresponding
        (x, y) offsets.
    """
    # Find insertion indices for _x and _y
    _x_ind = jnp.searchsorted(x_offsets, _x, side="left")
    _y_ind = jnp.searchsorted(y_offsets, _y, side="left")

    # Handle boundary conditions for x indices
    x_ind_low = jnp.clip(_x_ind - 1, 0, x_offsets.size - 1)
    x_ind_high = jnp.clip(_x_ind, 0, x_offsets.size - 1)

    # Handle boundary conditions for y indices
    y_ind_low = jnp.clip(_y_ind - 1, 0, y_offsets.size - 1)
    y_ind_high = jnp.clip(_y_ind, 0, y_offsets.size - 1)

    # Collect the two nearest indices for x and y
    x_inds = jnp.array([x_ind_low, x_ind_high])
    y_inds = jnp.array([y_ind_low, y_ind_high])

    # Manually create the four combinations without using meshgrid
    near_inds = jnp.array(
        [
            [x_inds[0], y_inds[0]],
            [x_inds[0], y_inds[1]],
            [x_inds[1], y_inds[0]],
            [x_inds[1], y_inds[1]],
        ]
    )

    # Extract the corresponding x and y offset values
    x_vals_low = x_offsets[x_inds[0]]
    x_vals_high = x_offsets[x_inds[1]]
    y_vals_low = y_offsets[y_inds[0]]
    y_vals_high = y_offsets[y_inds[1]]

    # Manually create the corresponding (x, y) offset combinations
    near_offsets = jnp.array(
        [
            [x_vals_low, y_vals_low],
            [x_vals_low, y_vals_high],
            [x_vals_high, y_vals_low],
            [x_vals_high, y_vals_high],
        ]
    )
    return near_inds, near_offsets


def shift_and_mask(near_psf, shift_x, shift_y, weight, x_grid, y_grid):
    """Shifts the PSF in x and y directions and applies a weight mask.

    Args:
        near_psf: The PSF image to shift, shape (height, width).
        shift_x: Shift in the x-direction, in pixels.
        shift_y: Shift in the y-direction, in pixels.
        weight: Weight for the PSF, used as the mask fill value.
        x_grid: The x pixel-index grid used to build the mask.
        y_grid: The y pixel-index grid used to build the mask.

    Returns:
        A tuple ``(weighted_psf, weight_mask)``: the shifted PSF multiplied by
        the mask, and the mask, whose valid pixels hold ``weight``.
    """
    # Shift the PSF in x and y directions
    shifted_psf = fft_shift_x(near_psf, shift_x)
    shifted_psf = fft_shift_y(shifted_psf, shift_y)

    # Create the weight mask
    weight_mask = create_shift_mask(
        near_psf, shift_x, shift_y, x_grid, y_grid, fill_val=weight
    )

    # Apply the weight mask
    weighted_psf = weight_mask * shifted_psf

    return weighted_psf, weight_mask


def _safe_hypot(x, y):
    """Return ``sqrt(x**2 + y**2)`` with a finite derivative at the origin.

    The value is unchanged; at the origin the radius has a cusp and the
    derivative is taken as zero instead of NaN.
    """
    r2 = x**2 + y**2
    positive = r2 > 0
    safe_radius = jnp.sqrt(jnp.where(positive, r2, like(1, r2)))
    return jnp.where(positive, safe_radius, like(0, r2))


def clamp_to_radius(x, y, radius):
    """Scale ``(x, y)`` toward the origin so it lies within ``radius``.

    The direction is kept and a point already inside is returned unchanged. A
    read beyond its tabulated positions then lands on the outermost ring,
    where the read and its derivative are finite, instead of extrapolating.
    The caller zeroes the result for a source outside; the gradient of the
    clamped read is finite, so the zero is an exact zero gradient.

    Args:
        x: X coordinate, in the unit of ``radius``.
        y: Y coordinate, in the unit of ``radius``.
        radius: Largest allowed distance from the origin.

    Returns:
        A tuple ``(x, y)`` of the scaled coordinates, each with the shape of
        the input.
    """
    r = _safe_hypot(x, y)
    outside = r > radius
    one = like(1, r)
    scale = jnp.where(outside, radius / jnp.where(outside, r, one), one)
    return x * scale, y * scale


def convert_xy_1D(x, y):
    """Converts x and y to 1D coordinates.

    Args:
        x: X coordinate.
        y: Y coordinate, in the unit of ``x``.

    Returns:
        A tuple ``(radius, 0)``: the radial separation of (x, y) with a finite
        derivative at the origin, and a zero y coordinate.
    """
    return _safe_hypot(x, y), 0


def convert_xy_2DQ(x, y):
    """Converts x and y to 2D quarter symmetric coordinates.

    Args:
        x: X coordinate.
        y: Y coordinate.

    Returns:
        A tuple ``(abs(x), abs(y))``, the coordinates folded into the first
        quadrant.
    """
    return jnp.abs(x), jnp.abs(y)


def convert_xy_2D(x, y):
    """Converts x and y to 2D coordinates.

    Args:
        x: X coordinate.
        y: Y coordinate.

    Returns:
        A tuple ``(x, y)``, unchanged.
    """
    return x, y


def basic_shift_val(input_val, converted_val, pixel_scale_arcsec):
    """Calculates the shift in pixels for a basic shift.

    Args:
        input_val: Requested coordinate.
        converted_val: Coordinate of the tabulated PSF, in the unit of
            ``input_val``.
        pixel_scale_arcsec: Pixel scale, in the unit of the coordinates per
            pixel.

    Returns:
        The shift, in pixels, from the tabulated position to the requested one.
    """
    return (input_val - converted_val) / pixel_scale_arcsec


def sym_shift_val(input_val, converted_val, pixel_scale_arcsec):
    """Calculates the shift in pixels for a symmetric shift.

    Args:
        input_val: Requested coordinate; its sign selects the mirrored side.
        converted_val: Coordinate of the tabulated PSF on the positive side, in
            the unit of ``input_val``.
        pixel_scale_arcsec: Pixel scale, in the unit of the coordinates per
            pixel.

    Returns:
        The shift, in pixels, from the signed tabulated position to the
        requested one.
    """
    sign = jnp.where(input_val >= 0, 1.0, -1.0)
    return (input_val - sign * converted_val) / pixel_scale_arcsec


def x_basic_shift(input_val, converted_val, PSF, pixel_scale_arcsec):
    """Shifts the PSF to the specified x position.

    Args:
        input_val: Requested x coordinate.
        converted_val: x coordinate of the tabulated PSF, in the unit of
            ``input_val``.
        PSF: The tabulated PSF image, shape (height, width).
        pixel_scale_arcsec: Pixel scale, in the unit of the coordinates per
            pixel.

    Returns:
        The PSF shifted along x, same shape as ``PSF``.
    """
    shift = basic_shift_val(input_val, converted_val, pixel_scale_arcsec)
    return fft_shift_x(PSF, shift)


def y_basic_shift(input_val, converted_val, PSF, pixel_scale_arcsec):
    """Shifts the PSF to the specified y position.

    Args:
        input_val: Requested y coordinate.
        converted_val: y coordinate of the tabulated PSF, in the unit of
            ``input_val``.
        PSF: The tabulated PSF image, shape (height, width).
        pixel_scale_arcsec: Pixel scale, in the unit of the coordinates per
            pixel.

    Returns:
        The PSF shifted along y, same shape as ``PSF``.
    """
    shift = basic_shift_val(input_val, converted_val, pixel_scale_arcsec)
    return fft_shift_y(PSF, shift)


def x_symmetric_shift(input_val, converted_val, PSF, pixel_scale_arcsec):
    """Shifts the PSF to the specified position assuming symmetry about x=0.

    Args:
        input_val: Requested x coordinate; a negative value mirrors the PSF.
        converted_val: x coordinate of the tabulated PSF on the positive side,
            in the unit of ``input_val``.
        PSF: The tabulated PSF image, shape (height, width).
        pixel_scale_arcsec: Pixel scale, in the unit of the coordinates per
            pixel.

    Returns:
        The PSF, mirrored when ``input_val`` is negative and shifted along
        x, same shape as ``PSF``.
    """
    # Apply a horizontal flip if the input value is negative
    _PSF = jnp.where(input_val < 0, jnp.fliplr(PSF), PSF)

    # Calculate the distance to shift the PSF
    shift = sym_shift_val(input_val, converted_val, pixel_scale_arcsec)
    # Apply the shift
    return fft_shift_x(_PSF, shift)


def y_symmetric_shift(input_val, converted_val, PSF, pixel_scale_arcsec):
    """Shifts the PSF to the specified position assuming symmetry about y=0.

    Args:
        input_val: Requested y coordinate; a negative value mirrors the PSF.
        converted_val: y coordinate of the tabulated PSF on the positive side,
            in the unit of ``input_val``.
        PSF: The tabulated PSF image, shape (height, width).
        pixel_scale_arcsec: Pixel scale, in the unit of the coordinates per
            pixel.

    Returns:
        The PSF, mirrored when ``input_val`` is negative and shifted along
        y, same shape as ``PSF``.
    """
    # Apply a vertical flip if the input value is negative
    _PSF = jnp.where(input_val < 0, jnp.flipud(PSF), PSF)

    # Get the distance to shift the PSF
    shift = sym_shift_val(input_val, converted_val, pixel_scale_arcsec)
    # Apply the shift
    return fft_shift_y(_PSF, shift)


def blend_pad(n_pixels):
    """Zero padding per side to limit periodic ringing in fractional shifts.

    The translated window mask excludes every periodic copy of an integer
    shift. For a fractional shift, the Fourier interpolation tails of those
    copies reach inside the window. In 256-pixel EAC1 reads with a bright
    truncated wing, reducing the pad from 384 to 372 changed a pixel by
    3.78e-5 of the read peak. Every smaller even FFT size with prime factors
    2, 3 and 5 also exceeded 1e-6 of the peak relative to the 384-pixel pad.

    Args:
        n_pixels: Number of pixels along one axis of the PSF.

    Returns:
        Pixels per side, rounded up. Symmetric padding preserves the parity
        of the stored length.
    """
    return (3 * n_pixels + 1) // 2


# Shifts within this many pixels of a half-pixel boundary round up to the next
# whole pixel, so a read and the scene operator split a shift the same way.
_FRACTION_TOL = 1e-5


def _whole_pixel_shift(shift):
    """Whole-pixel part of a translation in pixels, leaving at most half a pixel.

    Floor division keeps this valid for NumPy arrays on the host and traced
    JAX arrays alike, so the point reader and the scene operator place a
    stored PSF window at the same whole-pixel offset.
    """
    return (shift + 0.5 + _FRACTION_TOL) // 1


def _blend_windows(spec, weights, shift_x, shift_y, n_pad, shape):
    """Weighted sum of shifted neighbors, each zero outside its stored window.

    A stored PSF carries no data beyond its array, so each translated neighbor
    contributes only inside its stored window moved by the whole-pixel part of
    its shift. Each neighbor is inverse transformed on its own: summing the
    spectra first would leave the Fourier ringing of every truncated PSF in
    pixels its window does not reach.

    Args:
        spec: Shifted neighbor spectra [K, H_pad, W_pad // 2 + 1], FFT along y
            and real FFT along x.
        weights: Blend weight of each neighbor [K].
        shift_x: Shift of each neighbor along x, in pixels [K].
        shift_y: Shift of each neighbor along y, in pixels [K].
        n_pad: Zero padding added to each side of both axes, in pixels.
        shape: Detector shape (H, W) of one stored PSF.

    Returns:
        The blended image on the stored PSF grid, shape (H, W).
    """
    h, w = shape
    w_pad = w + 2 * n_pad
    # Rows are cut before the x transform: each row's inverse is independent.
    planes = jnp.fft.ifft(spec, axis=1)[:, n_pad : n_pad + h, :]
    planes = jnp.fft.irfft(planes, n=w_pad, axis=2)[:, :, n_pad : n_pad + w]
    rows = (
        jnp.arange(h, dtype=shift_y.dtype)[None, :, None]
        - _whole_pixel_shift(shift_y)[:, None, None]
    )
    cols = (
        jnp.arange(w, dtype=shift_x.dtype)[None, None, :]
        - _whole_pixel_shift(shift_x)[:, None, None]
    )
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    return jnp.sum(
        jnp.where(inside, planes, like(0, planes)) * weights[:, None, None], axis=0
    )


def separable_read_plan(
    x,
    y,
    pixel_scale_arcsec,
    x_offsets,
    y_offsets,
    offset_to_flat_idx,
    x_symmetric,
    y_symmetric,
    input_type,
):
    """Return the nodes, weights and pixel shifts of one separable blend read.

    The result is ``(flat_indices, weights, shift_x, shift_y)``, one entry per
    neighbor. A PSF read is the weighted sum of the neighbors, each translated by
    its shift in the canonical quadrant, then reflected for negative coordinates
    when the response is symmetric.

    Args:
        x: X position of the source, in the same unit as ``x_offsets``.
        y: Y position of the source, in the same unit as ``y_offsets``.
        pixel_scale_arcsec: Pixel scale, in the unit of the offsets per pixel.
        x_offsets: Sorted 1D array of unique tabulated x offsets.
        y_offsets: Sorted 1D array of unique tabulated y offsets.
        offset_to_flat_idx: 2D integer array mapping (x index, y index) to the
            index of the PSF in the flat PSF array.
        x_symmetric: Whether the response is symmetric about x=0.
        y_symmetric: Whether the response is symmetric about y=0.
        input_type: Layout of the tabulated offsets, "1d" or "2dq"; any other
            value is read with the 2D quarter lookup.

    Returns:
        A tuple ``(flat_indices, weights, shift_x, shift_y)``, each with one
        entry per neighbor (2 for "1d", 4 otherwise): the flat PSF indices,
        the interpolation weights, and the pixel shifts along x and y.
    """
    # Neighbor lookup
    if input_type == "1d":
        _x, _y = convert_xy_1D(x, y)
        inds, offsets = get_near_inds_offsets_1D(x_offsets, y_offsets, _x, _y)
    else:
        _x, _y = convert_xy_2DQ(x, y)
        inds, offsets = get_near_inds_offsets_2D(x_offsets, y_offsets, _x, _y)

    # Use the index mapping to get flat_psfs indices from (x_idx, y_idx) pairs
    flat_indices = offset_to_flat_idx[inds[:, 0], inds[:, 1]]

    # We calculate weights based on the relative position of _x/_y
    # within the bounding box of the neighbors found.
    if input_type == "1d":
        # Neighbors are: [x_low, 0], [x_high, 0]
        x0 = offsets[0, 0]
        x1 = offsets[1, 0]

        # Span of the current grid cell
        span_x = x1 - x0

        # Avoid division by zero if query lands exactly on a node or grid is 1 point
        span_x = jnp.where(span_x == 0, like(1, span_x), span_x)

        # Normalize distance (0.0 to 1.0)
        u = (_x - x0) / span_x

        # Linear weights: w0 = (1-u), w1 = u
        # shape (2,)
        weights = jnp.array([1.0 - u, u])

        # Safety: If span was 0, force weight to [1, 0]
        weights = jnp.where(x1 == x0, jnp.array([1.0, 0.0], weights.dtype), weights)

    else:  # 2D Case
        # Neighbors order from get_near_inds_offsets_2D is:
        # 0: (x0, y0), 1: (x0, y1), 2: (x1, y0), 3: (x1, y1)
        x0 = offsets[0, 0]
        x1 = offsets[3, 0]  # dim 0 is x
        y0 = offsets[0, 1]
        y1 = offsets[3, 1]  # dim 1 is y

        span_x = x1 - x0
        span_y = y1 - y0

        # Avoid div by zero
        span_x = jnp.where(span_x == 0, like(1, span_x), span_x)
        span_y = jnp.where(span_y == 0, like(1, span_y), span_y)

        # Normalize coordinates (0.0 to 1.0)
        u = (_x - x0) / span_x
        v = (_y - y0) / span_y

        # Bilinear Weights
        # w00 = (1-u)(1-v)
        # w01 = (1-u)v
        # w10 = u(1-v)
        # w11 = uv
        w0 = (1.0 - u) * (1.0 - v)
        w1 = (1.0 - u) * v
        w2 = u * (1.0 - v)
        w3 = u * v

        weights = jnp.array([w0, w1, w2, w3])

        # Safety for coincident nodes
        weights = jnp.where(
            (x1 == x0) & (y1 == y0),
            jnp.array([1.0, 0.0, 0.0, 0.0], weights.dtype),
            weights,
        )

    # Shift in the canonical quadrant; reflect only after interpolation so
    # fractional Nyquist shifts agree with quarter-datacube reconstruction.
    query_x = jnp.where(x < 0, -x, x) if x_symmetric else x
    query_y = jnp.where(y < 0, -y, y) if y_symmetric else y
    shift_x = (query_x - offsets[:, 0]) / pixel_scale_arcsec
    shift_y = (query_y - offsets[:, 1]) / pixel_scale_arcsec
    return flat_indices, weights, shift_x, shift_y


def synthesize_psf_separable(
    x,
    y,
    pixel_scale_arcsec,
    flat_psfs,
    x_offsets,
    y_offsets,
    offset_to_flat_idx,
    kx,
    ky,
    n_pad,
    x_symmetric,
    y_symmetric,
    input_type,
):
    """Synthesizes PSF using separable 1D FFTs.

    Optimized for speed and stability with even-sized padding.
    Uses flat_psfs with offset_to_flat_idx mapping for memory efficiency.
    Each neighbor contributes only inside its stored window translated by the
    whole-pixel part of its shift; beyond it the table has no data.

    For 1D input the neighbors are the tabulated PSFs along +x at the radial
    separation of (x, y). They are mirrored for x < 0 or y < 0 and translated,
    but not rotated to the position angle: rotation would also rotate the
    pupil's diffraction structure, which is fixed on the detector.

    Args:
        x:
            X coordinate in lambda/D.
        y:
            Y coordinate in lambda/D.
        pixel_scale_arcsec:
            Pixel scale in lambda/D per pixel.
        flat_psfs:
            Flat array of PSFs with shape (N_psfs, H, W).
        x_offsets:
            Sorted array of unique x offsets.
        y_offsets:
            Sorted array of unique y offsets.
        offset_to_flat_idx:
            2D array mapping (x_idx, y_idx) -> flat_psfs index.
        kx:
            FFT frequencies for x-axis.
        ky:
            FFT frequencies for y-axis.
        n_pad:
            Number of padding pixels.
        x_symmetric:
            Whether PSFs are symmetric about x=0.
        y_symmetric:
            Whether PSFs are symmetric about y=0.
        input_type:
            Type of input data ("1d", "2dq", or "2df").

    Returns:
        Synthesized PSF at the requested (x, y) position, shape (H, W).
    """
    flat_indices, weights, shift_x, shift_y = separable_read_plan(
        x,
        y,
        pixel_scale_arcsec,
        x_offsets,
        y_offsets,
        offset_to_flat_idx,
        x_symmetric,
        y_symmetric,
        input_type,
    )
    neighbors = flat_psfs[flat_indices]

    # Pad (using consistent n_pad from init)
    # (K, H, W) -> (K, H_pad, W_pad)
    pad_width = ((0, 0), (n_pad, n_pad), (n_pad, n_pad))
    neighbors = jnp.pad(neighbors, pad_width)

    # FFT X (Real-to-Complex, Axis 2)
    # Output: (K, H_pad, W_pad//2 + 1)
    spec = jnp.fft.rfft(neighbors, axis=2)

    # Apply X-Shift
    # kx: (W_freq,). shift_x: (K,).
    # Broadcast: (K, 1, W_freq)
    phase_x = -2 * jnp.pi * shift_x[:, None, None] * kx[None, None, :]
    spec = spec * jnp.exp(1j * phase_x)

    # FFT Y (Complex-to-Complex, Axis 1)
    spec = jnp.fft.fft(spec, axis=1)

    # Apply Y-Shift
    # ky: (H_pad,). shift_y: (K,).
    # Broadcast: (K, H_pad, 1)
    phase_y = -2 * jnp.pi * shift_y[:, None, None] * ky[None, :, None]
    spec = spec * jnp.exp(1j * phase_y)

    psf = _blend_windows(spec, weights, shift_x, shift_y, n_pad, flat_psfs.shape[-2:])

    if x_symmetric:
        psf = lax.cond(x < 0, lambda p: jnp.flip(p, axis=1), lambda p: p, psf)
    if y_symmetric:
        psf = lax.cond(y < 0, lambda p: jnp.flip(p, axis=0), lambda p: p, psf)

    psf = jnp.maximum(psf, 0.0)

    return psf


# Nodes whose radii differ by less than this fraction of (1 + radius) share a
# ring. It is wide enough for the single-precision rounding of the radii (about
# 4e-6 lambda/D at 60) and far below any gap between rings.
_RING_TOL = 1e-5


def _angular_bracket(theta_q, theta_n, member):
    """The two member nodes that bracket an angle, with their weights.

    Returns ``(ahead, behind, w_ahead)``: the member node first reached turning
    counter-clockwise from ``theta_q``, the one first reached turning clockwise,
    and the linear weight of the first (the second gets ``1 - w_ahead``). The
    angle is periodic. A ring with one member returns it twice.
    """
    delta = theta_n - theta_q
    inf = like(jnp.inf, delta)
    turn = like(2 * jnp.pi, delta)
    d_ahead = jnp.where(member, delta % turn, inf)
    d_behind = jnp.where(member, -delta % turn, inf)
    ahead, behind = jnp.argmin(d_ahead), jnp.argmin(d_behind)
    span = d_ahead[ahead] + d_behind[behind]
    w_ahead = d_behind[behind] / jnp.where(span > 0, span, like(1, span))
    return ahead, behind, jnp.where(span > 0, w_ahead, like(0.5, w_ahead))


def idw_read_plan(
    x,
    y,
    pixel_scale_arcsec,
    flat_x_offsets,
    flat_y_offsets,
    x_symmetric,
    y_symmetric,
    input_type,
):
    """Return the nodes, weights and pixel shifts of one polar blend read.

    The result is ``(indices, weights, shift_x, shift_y)``, four entries each. A
    PSF read is the weighted sum of the four nodes, each translated by its
    shift in the canonical quadrant, then reflected for negative coordinates
    when the response is symmetric.

    The weights are bilinear in radius and azimuth. Nodes of one radius form a
    ring; the read takes the two rings that bracket the query radius, linear in
    radius, and in each ring the two nodes that bracket the query azimuth,
    linear in angle. Both interpolations are linear, so the read is continuous
    in position, reproduces a node exactly at the node, and follows a radial
    ramp such as the throughput at the inner working angle instead of
    saturating at the nearest node. Beyond the first or last ring the nearest
    ring is used alone. A ring of one node (the origin) ignores the azimuth.

    The weights sum to one and the read is a PSF, not a flux-normalized image:
    its total is the bilinear interpolation of the node totals.

    Args:
        x: X position of the source, in the same unit as ``flat_x_offsets``.
        y: Y position of the source, in the same unit as ``flat_y_offsets``.
        pixel_scale_arcsec: Pixel scale, in the unit of the offsets per pixel.
        flat_x_offsets: x offsets of all tabulated nodes, one per node.
        flat_y_offsets: y offsets of all tabulated nodes, one per node.
        x_symmetric: Whether the response is symmetric about x=0.
        y_symmetric: Whether the response is symmetric about y=0.
        input_type: Layout of the tabulated offsets: "2dq" folds the query
            into the first quadrant, "1d" uses its radial separation, and any
            other value uses (x, y) unchanged.

    Returns:
        A tuple ``(indices, weights, shift_x, shift_y)``, each of shape (4,):
        the node indices, the bilinear weights, and the pixel shifts along x
        and y.
    """
    # Handle Symmetry (Map to 1st Quadrant if 2DQ)
    if input_type == "2dq":
        _x, _y = convert_xy_2DQ(x, y)
    elif input_type == "1d":
        _x, _y = convert_xy_1D(x, y)
    else:
        _x, _y = x, y

    # The radius is a square root that vanishes at the origin, so it goes
    # through _safe_hypot to keep the gradient finite there.
    r_q = _safe_hypot(_x, _y)
    at_origin = r_q == 0
    theta_q = jnp.arctan2(_y, jnp.where(at_origin, like(1, _x), _x))
    r_n = jnp.hypot(flat_x_offsets, flat_y_offsets)
    theta_n = jnp.arctan2(flat_y_offsets, flat_x_offsets)

    # Bracketing ring radii; a missing side repeats the other, giving t = 0.
    inf = like(jnp.inf, r_n)
    below = r_n <= r_q + _RING_TOL * (1.0 + r_q)
    r_lo = jnp.max(jnp.where(below, r_n, -inf))
    r_hi = jnp.min(jnp.where(below, inf, r_n))
    r_lo = jnp.where(jnp.isfinite(r_lo), r_lo, r_hi)
    r_hi = jnp.where(jnp.isfinite(r_hi), r_hi, r_lo)
    gap = r_hi - r_lo
    safe_gap = jnp.where(gap > 0, gap, like(1, gap))
    t = jnp.where(gap > 0, jnp.clip((r_q - r_lo) / safe_gap, 0, 1), like(0, gap))

    inds, weights = [], []
    for ring_r, ring_w in ((r_lo, 1.0 - t), (r_hi, t)):
        ahead, behind, w_ahead = _angular_bracket(
            theta_q, theta_n, jnp.abs(r_n - ring_r) <= _RING_TOL * (1.0 + ring_r)
        )
        inds += [ahead, behind]
        weights += [ring_w * w_ahead, ring_w * (1.0 - w_ahead)]
    inds, weights = jnp.stack(inds), jnp.stack(weights)

    # Interpolate in the canonical quadrant, then reflect the finished image.
    # Reflecting before a fractional FFT shift can disagree at the Nyquist
    # boundary; post-interpolation reflection makes quarter storage exact.
    query_x = jnp.where(x < 0, -x, x) if x_symmetric else x
    query_y = jnp.where(y < 0, -y, y) if y_symmetric else y
    shift_x = (query_x - flat_x_offsets[inds]) / pixel_scale_arcsec
    shift_y = (query_y - flat_y_offsets[inds]) / pixel_scale_arcsec
    return inds, weights, shift_x, shift_y


def synthesize_psf_idw(
    x,
    y,
    pixel_scale_arcsec,
    flat_psfs,
    flat_x_offsets,
    flat_y_offsets,
    kx,
    ky,
    n_pad,
    x_symmetric,
    y_symmetric,
    input_type,
):
    """Synthesize a PSF from polar-ring nodes by bilinear (radius, azimuth) blending.

    See ``idw_read_plan`` for the weights. Each neighbor contributes only
    inside its stored window translated by the whole-pixel part of its shift;
    beyond it the table has no data. The result is a PSF, not a
    flux-normalized image.

    Args:
        x: X position of the source, in the same unit as ``flat_x_offsets``.
        y: Y position of the source, in the same unit as ``flat_y_offsets``.
        pixel_scale_arcsec: Pixel scale, in the unit of the offsets per pixel.
        flat_psfs: Tabulated PSFs, shape (N_psfs, H, W).
        flat_x_offsets: x offsets of the nodes, one per PSF.
        flat_y_offsets: y offsets of the nodes, one per PSF.
        kx: FFT frequencies along x for the padded real FFT.
        ky: FFT frequencies along y for the padded FFT.
        n_pad: Zero padding added to each side of both axes, in pixels.
        x_symmetric: Whether the response is symmetric about x=0.
        y_symmetric: Whether the response is symmetric about y=0.
        input_type: Layout of the tabulated offsets ("1d", "2dq" or other),
            as in ``idw_read_plan``.

    Returns:
        The synthesized PSF at (x, y), shape (H, W), clipped to be
        non-negative.
    """
    inds, weights, shift_x, shift_y = idw_read_plan(
        x,
        y,
        pixel_scale_arcsec,
        flat_x_offsets,
        flat_y_offsets,
        x_symmetric,
        y_symmetric,
        input_type,
    )
    neighbors = flat_psfs[inds]

    # Pad and FFT (Standard Shift Logic)
    pad_width = ((0, 0), (n_pad, n_pad), (n_pad, n_pad))
    neighbors = jnp.pad(neighbors, pad_width)

    # FFT X
    spec = jnp.fft.rfft(neighbors, axis=2)
    phase_x = -2 * jnp.pi * shift_x[:, None, None] * kx[None, None, :]
    spec = spec * jnp.exp(1j * phase_x)

    # FFT Y
    spec = jnp.fft.fft(spec, axis=1)
    phase_y = -2 * jnp.pi * shift_y[:, None, None] * ky[None, :, None]
    spec = spec * jnp.exp(1j * phase_y)

    psf = _blend_windows(spec, weights, shift_x, shift_y, n_pad, flat_psfs.shape[-2:])

    if x_symmetric:
        psf = lax.cond(x < 0, lambda p: jnp.flip(p, axis=1), lambda p: p, psf)
    if y_symmetric:
        psf = lax.cond(y < 0, lambda p: jnp.flip(p, axis=0), lambda p: p, psf)
    return jnp.maximum(psf, 0.0)
