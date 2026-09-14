"""JAX functions for image processing operations."""

from functools import partial

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
from hwoutils.transforms import rotate_image
from jax import lax


def create_shift_mask(psf, shift_x, shift_y, x_grid, y_grid, fill_val=1):
    """Create a mask to identify valid pixels to average.

    This function is useful because when the PSF is shifted there are empty
    pixels, since they were outside the initial image, and should not be
    included in the final average. This version avoids lax.cond for better
    performance under JIT.

    Args:
        psf (jax.numpy.ndarray):
            The PSF image to shift.
        shift_x (float):
            The shift in the x direction.
        shift_y (float):
            The shift in the y direction.
        x_grid (jax.numpy.ndarray):
            The x-coordinate grid.
        y_grid (jax.numpy.ndarray):
            The y-coordinate grid.
        fill_val (float, optional):
            The value to fill the mask with.

    Returns:
        jax.numpy.ndarray:
            The mask to identify valid pixels to average.
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
        x_offsets (jnp.ndarray): 1D array of x offsets, sorted in ascending order.
        y_offsets (jnp.ndarray): 1D array of y offsets.
        _x (float): The x-coordinate for which to find nearby offsets.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            - near_inds: Array of shape (2,) containing the indices of the two
              surrounding points.
            - near_offsets: Array of shape (2,) containing the corresponding x offsets.
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
    x_phasor: jnp.ndarray,
    y_phasor: jnp.ndarray,
    reshaped_psfs: jnp.ndarray,
):
    """Creates and returns the PSF at the specified off-axis position using JAX."""
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
        x_phasor,
        y_phasor,
    )
    shifted_psf2, mask2 = shift_and_mask(
        near_psfs[1],
        near_shifts[1, 0],
        near_shifts[1, 1],
        weights[1],
        x_grid,
        y_grid,
        x_phasor,
        y_phasor,
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
    x_phasor: jnp.ndarray,
    y_phasor: jnp.ndarray,
    reshaped_psfs: jnp.ndarray,
):
    """Creates and returns the PSF at the specified off-axis position using JAX."""
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
        x_phasor,
        y_phasor,
    )
    shifted_psf2, mask2 = shift_and_mask(
        near_psfs[1],
        near_shifts[1, 0],
        near_shifts[1, 1],
        weights[1],
        x_grid,
        y_grid,
        x_phasor,
        y_phasor,
    )
    shifted_psf3, mask3 = shift_and_mask(
        near_psfs[2],
        near_shifts[2, 0],
        near_shifts[2, 1],
        weights[2],
        x_grid,
        y_grid,
        x_phasor,
        y_phasor,
    )
    shifted_psf4, mask4 = shift_and_mask(
        near_psfs[3],
        near_shifts[3, 0],
        near_shifts[3, 1],
        weights[3],
        x_grid,
        y_grid,
        x_phasor,
        y_phasor,
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
        x_offsets (jnp.ndarray): 1D array of x offsets, sorted in ascending order.
        y_offsets (jnp.ndarray): 1D array of y offsets, sorted in ascending order.
        _x (float): The x-coordinate for which to find nearby offsets.
        _y (float): The y-coordinate for which to find nearby offsets.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            - near_inds: Array of shape (4, 2) containing the indices of the
              four surrounding points.
            - near_offsets: Array of shape (4, 2) containing the corresponding
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


def shift_and_mask(
    near_psf, shift_x, shift_y, weight, x_grid, y_grid, x_phasor, y_phasor
):
    """Shifts the PSF in x and y directions and applies a weight mask.

    Args:
        near_psf (jax.numpy.ndarray): The PSF image to shift.
        shift_x (float): Shift in the x-direction.
        shift_y (float): Shift in the y-direction.
        weight (float): Weight for the PSF.
        x_grid (jax.numpy.ndarray): The x-coordinate grid.
        y_grid (jax.numpy.ndarray): The y-coordinate grid.
        x_phasor (jax.numpy.ndarray): Precomputed components for the Fourier shift.
        y_phasor (jax.numpy.ndarray): Precomputed components for the Fourier shift.

    Returns:
        Tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
            - Weighted shifted PSF.
            - Weight mask.
    """
    # Shift the PSF in x and y directions
    shifted_psf = fft_shift_x(near_psf, shift_x, x_phasor)
    shifted_psf = fft_shift_y(shifted_psf, shift_y, y_phasor)

    # Create the weight mask
    weight_mask = create_shift_mask(
        near_psf, shift_x, shift_y, x_grid, y_grid, fill_val=weight
    )

    # Apply the weight mask
    weighted_psf = weight_mask * shifted_psf

    return weighted_psf, weight_mask


def convert_xy_1D(x, y):
    """Converts x and y to 1D coordinates."""
    return jnp.sqrt(x**2 + y**2), 0


def convert_xy_2DQ(x, y):
    """Converts x and y to 2D quarter symmetric coordinates."""
    return jnp.abs(x), jnp.abs(y)


def convert_xy_2D(x, y):
    """Converts x and y to 2D coordinates."""
    return x, y


def basic_shift_val(input_val, converted_val, pixel_scale_arcsec):
    """Calculates the shift in pixels for a basic shift."""
    return (input_val - converted_val) / pixel_scale_arcsec


def sym_shift_val(input_val, converted_val, pixel_scale_arcsec):
    """Calculates the shift in pixels for a symmetric shift."""
    sign = jnp.where(input_val >= 0, 1.0, -1.0)
    return (input_val - sign * converted_val) / pixel_scale_arcsec


def x_basic_shift(input_val, converted_val, PSF, pixel_scale_arcsec, x_phasor):
    """Shifts the PSF to the specified x position."""
    shift = basic_shift_val(input_val, converted_val, pixel_scale_arcsec)
    return fft_shift_x(PSF, shift, x_phasor)


def y_basic_shift(input_val, converted_val, PSF, pixel_scale_arcsec, y_phasor):
    """Shifts the PSF to the specified y position."""
    shift = basic_shift_val(input_val, converted_val, pixel_scale_arcsec)
    return fft_shift_y(PSF, shift, y_phasor)


def x_symmetric_shift(input_val, converted_val, PSF, pixel_scale_arcsec, x_phasor):
    """Shifts the PSF to the specified position assuming symmetry about x=0."""
    # Apply a horizontal flip if the input value is negative
    _PSF = jnp.where(input_val < 0, jnp.fliplr(PSF), PSF)

    # Calculate the distance to shift the PSF
    shift = sym_shift_val(input_val, converted_val, pixel_scale_arcsec)
    # Apply the shift
    return fft_shift_x(_PSF, shift, x_phasor)


def y_symmetric_shift(input_val, converted_val, PSF, pixel_scale_arcsec, y_phasor):
    """Shifts the PSF to the specified position assuming symmetry about y=0."""
    # Apply a vertical flip if the input value is negative
    _PSF = jnp.where(input_val < 0, jnp.flipud(PSF), PSF)

    # Get the distance to shift the PSF
    shift = sym_shift_val(input_val, converted_val, pixel_scale_arcsec)
    # Apply the shift
    return fft_shift_y(_PSF, shift, y_phasor)


def fft_rotate_jax(image, rot_deg, *, pad_factor=0.5, center=None):
    """Rotate a square image with the shared hwoutils Fourier implementation.

    Args:
        image: 2D square image.
        rot_deg: CCW angle in degrees when displayed with origin at lower left.
        pad_factor: Padding per side as a fraction of N. Default 0.5 gives
            approximately 2N width; use 1.5 for 4N and check convergence for
            precision wings or substantial edge content.
        center: Optional optical center (row, column), in input pixels.
            Defaults to the geometric array center (N-1)/2.

    Returns:
        Rotated image, without clipping or flux renormalization. See
        ``hwoutils.transforms.rotate_image`` for Nyquist and derivative semantics.
    """
    return rotate_image(image, rot_deg, pad_factor=pad_factor, center=center)


def rotate_with_shear(image, rot_deg, *, pad_factor=0.5):
    """Retain the legacy clockwise residual-angle convention using shared code.

    Args:
        image: 2D square image.
        rot_deg: Clockwise residual angle in degrees, normally in (-45, 45].
        pad_factor: Padding per side as a fraction of input width.

    Returns:
        Rotated image with the same shape as the input.
    """
    return rotate_image(image, -rot_deg, pad_factor=pad_factor)


def decompose_angle_jax(angle):
    """Decompose an angle from [0, 360) to (-45, 45] and 90 degree rotations.

    Args:
        angle (float):
            The input angle in degrees.

    Returns:
        tuple:
            - float: The rotation angle in the range (-45, 45] degrees.
            - int: The number of 90-degree rotations to apply.
    """
    # Normalize the angle to [0, 360)
    angle = angle % 360

    # Determine the number of 90-degree rotations
    n_rot = (angle // 90).astype(int)

    # Cut the angle to [0, 90)
    adjusted_angle = angle % 90

    # Adjust the angle to the range (-45, 45]
    adjusted_angle, n_rot = lax.cond(
        adjusted_angle > 45,
        lambda x: (x - 90, n_rot + 1),
        lambda x: (x, n_rot),
        adjusted_angle,
    )
    # if n_rot is 4, set it to 0 to avoid unnecessary rotations
    # this occurs when the angle is in the range (315, 360)
    n_rot = lax.cond(
        n_rot == 4,
        lambda x: 0,
        lambda x: x,
        n_rot,
    )

    return adjusted_angle, n_rot


def rot90_traceable(m, k=1, axes=(0, 1)):
    """Rotate an array by 90 degrees in the plane specified by axes.

    This function is a traceable version of `numpy.rot90` taken from the
    jax GitHub issues.

    Args:
        m (jax.numpy.ndarray):
            The input array to be rotated.
        k (int, optional):
            The number of 90-degree rotations to apply.
        axes (tuple, optional):
            The axes to rotate the array in.

    Returns:
        jax.numpy.ndarray:
            The rotated array.
    """
    k %= 4
    return lax.switch(k, [partial(jnp.rot90, m, k=i, axes=axes) for i in range(4)])


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
    max_offset,
):
    """Synthesizes PSF using separable 1D FFTs.

    Optimized for speed and stability with even-sized padding.
    Uses flat_psfs with offset_to_flat_idx mapping for memory efficiency.

    Args:
        x (float):
            X coordinate in lambda/D.
        y (float):
            Y coordinate in lambda/D.
        pixel_scale_arcsec (float):
            Pixel scale in lambda/D per pixel.
        flat_psfs (jnp.ndarray):
            Flat array of PSFs with shape (N_psfs, H, W).
        x_offsets (jnp.ndarray):
            Sorted array of unique x offsets.
        y_offsets (jnp.ndarray):
            Sorted array of unique y offsets.
        offset_to_flat_idx (jnp.ndarray):
            2D array mapping (x_idx, y_idx) -> flat_psfs index.
        kx (jnp.ndarray):
            FFT frequencies for x-axis.
        ky (jnp.ndarray):
            FFT frequencies for y-axis.
        n_pad (int):
            Number of padding pixels.
        x_symmetric (bool):
            Whether PSFs are symmetric about x=0.
        y_symmetric (bool):
            Whether PSFs are symmetric about y=0.
        input_type (str):
            Type of input data ("1d", "2dq", or "2df").
        max_offset (float):
            Maximum valid offset distance (-1 to disable bounds check).

    Returns:
        jnp.ndarray:
            Synthesized PSF at the requested (x, y) position.
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
    neighbors = flat_psfs[flat_indices]

    # We calculate weights based on the relative position of _x/_y
    # within the bounding box of the neighbors found.
    if input_type == "1d":
        # Neighbors are: [x_low, 0], [x_high, 0]
        x0 = offsets[0, 0]
        x1 = offsets[1, 0]

        # Span of the current grid cell
        span_x = x1 - x0

        # Avoid division by zero if query lands exactly on a node or grid is 1 point
        span_x = jnp.where(span_x == 0, 1.0, span_x)

        # Normalize distance (0.0 to 1.0)
        u = (_x - x0) / span_x

        # Linear weights: w0 = (1-u), w1 = u
        # shape (2,)
        weights = jnp.array([1.0 - u, u])

        # Safety: If span was 0, force weight to [1, 0]
        weights = jnp.where(x1 == x0, jnp.array([1.0, 0.0]), weights)

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
        span_x = jnp.where(span_x == 0, 1.0, span_x)
        span_y = jnp.where(span_y == 0, 1.0, span_y)

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
            (x1 == x0) & (y1 == y0), jnp.array([1.0, 0.0, 0.0, 0.0]), weights
        )

    # Symmetry logic
    eff_offsets_x = offsets[:, 0]
    eff_offsets_y = offsets[:, 1]

    if x_symmetric:
        eff_offsets_x = jnp.where(x < 0, -offsets[:, 0], offsets[:, 0])
        neighbors = lax.cond(
            x < 0, lambda p: jnp.flip(p, axis=2), lambda p: p, neighbors
        )
    if y_symmetric:
        eff_offsets_y = jnp.where(y < 0, -offsets[:, 1], offsets[:, 1])
        neighbors = lax.cond(
            y < 0, lambda p: jnp.flip(p, axis=1), lambda p: p, neighbors
        )

    # Calculate Shift (in pixels)
    shift_x = (x - eff_offsets_x) / pixel_scale_arcsec
    shift_y = (y - eff_offsets_y) / pixel_scale_arcsec

    # Pad (using consistent n_pad from init)
    # (K, H, W) -> (K, H_pad, W_pad)
    pad_width = ((0, 0), (n_pad, n_pad), (n_pad, n_pad))
    neighbors = jnp.pad(neighbors, pad_width)

    # Capture width for robust reconstruction
    _, _h_pad, w_pad = neighbors.shape

    # FFT X (Real-to-Complex, Axis 2)
    # Output: (K, H_pad, W_pad//2 + 1)
    spec = jnp.fft.rfft(neighbors, axis=2)

    # Apply X-Shift
    # kx: (W_freq,). shift_x: (K,).
    # Broadcast: (K, 1, W_freq)
    phasor_x = jnp.exp(-2j * jnp.pi * shift_x[:, None, None] * kx[None, None, :])
    spec = spec * phasor_x

    # FFT Y (Complex-to-Complex, Axis 1)
    spec = jnp.fft.fft(spec, axis=1)

    # Apply Y-Shift
    # ky: (H_pad,). shift_y: (K,).
    # Broadcast: (K, H_pad, 1)
    phasor_y = jnp.exp(-2j * jnp.pi * shift_y[:, None, None] * ky[None, :, None])
    spec = spec * phasor_y

    # Weighted Sum (Collapse Neighbors)
    # (K, H_freq, W_freq) -> (H_freq, W_freq)
    summed_spec = jnp.sum(spec * weights[:, None, None], axis=0)

    # IFFT Y
    res = jnp.fft.ifft(summed_spec, axis=0)

    # IFFT X (Complex-to-Real)
    # Pass n=w_pad to ensure we reconstruct to the exact even size
    res = jnp.fft.irfft(res, n=w_pad, axis=1)

    # Unpad
    h_orig = flat_psfs.shape[-2]
    psf = res[n_pad : n_pad + h_orig, n_pad : n_pad + h_orig]

    psf = jnp.maximum(psf, 0.0)

    # Mask if out of bounds
    # Only currently implemented for 1D (radial symmetry)
    # max_offset must be provided (> 0) to enable masking
    psf = lax.cond(
        (input_type == "1d") & (max_offset > 0),
        lambda p: jnp.where(jnp.sqrt(x**2 + y**2) > max_offset, 0.0, p),
        lambda p: p,
        psf,
    )

    return psf


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
    k_neighbors=4,
):
    """Synthesizes PSF using Power-2 Polar IDW for irregular grids.

    Distance is the tangent-plane arc length at the query,
    ``d = sqrt(dr**2 + (r_q * dtheta)**2)``, and weights use ``1 / d**2``.
    The polar metric matches the radial/azimuthal anisotropy of coronagraph
    PSF structure, and the squared exponent acts as a soft nearest-neighbor
    on smooth grids. See yippy-paper Section 4 for the kernel-selection
    rationale.
    """
    # Handle Symmetry (Map to 1st Quadrant if 2DQ)
    if input_type == "2dq":
        _x, _y = convert_xy_2DQ(x, y)
    elif input_type == "1d":
        _x, _y = convert_xy_1D(x, y)
    else:
        _x, _y = x, y

    # Polar arc-length distance, linearized at the query.
    r_q = jnp.hypot(_x, _y)
    theta_q = jnp.arctan2(_y, _x)
    r_n = jnp.hypot(flat_x_offsets, flat_y_offsets)
    theta_n = jnp.arctan2(flat_y_offsets, flat_x_offsets)
    dr = r_n - r_q
    dtheta = (theta_n - theta_q + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    arc = r_q * dtheta
    dists = jnp.sqrt(dr**2 + arc**2)

    # Find K Nearest Neighbors
    # We use top_k on negative distance to find the smallest distances
    neg_dists, inds = lax.top_k(-dists, k_neighbors)
    near_dists = -neg_dists

    # Get Neighbors
    neighbors = flat_psfs[inds]

    # Power-2 inverse-distance weights with a small epsilon to avoid /0.
    weights = 1.0 / (near_dists + 1e-10) ** 2
    weights = weights / jnp.sum(weights)

    # Apply Symmetry Flips to neighbors
    eff_offsets_x = flat_x_offsets[inds]
    eff_offsets_y = flat_y_offsets[inds]

    if x_symmetric:
        # Check against original x
        neighbors = lax.cond(
            x < 0, lambda p: jnp.flip(p, axis=2), lambda p: p, neighbors
        )
        eff_offsets_x = jnp.where(x < 0, -eff_offsets_x, eff_offsets_x)

    if y_symmetric:
        # Check against original y
        neighbors = lax.cond(
            y < 0, lambda p: jnp.flip(p, axis=1), lambda p: p, neighbors
        )
        eff_offsets_y = jnp.where(y < 0, -eff_offsets_y, eff_offsets_y)

    # Calculate Shift (in pixels)
    shift_x = (x - eff_offsets_x) / pixel_scale_arcsec
    shift_y = (y - eff_offsets_y) / pixel_scale_arcsec

    # Pad and FFT (Standard Shift Logic)
    pad_width = ((0, 0), (n_pad, n_pad), (n_pad, n_pad))
    neighbors = jnp.pad(neighbors, pad_width)

    # FFT X
    spec = jnp.fft.rfft(neighbors, axis=2)
    phasor_x = jnp.exp(-2j * jnp.pi * shift_x[:, None, None] * kx[None, None, :])
    spec = spec * phasor_x

    # FFT Y
    spec = jnp.fft.fft(spec, axis=1)
    phasor_y = jnp.exp(-2j * jnp.pi * shift_y[:, None, None] * ky[None, :, None])
    spec = spec * phasor_y

    # Weighted Sum
    summed_spec = jnp.sum(spec * weights[:, None, None], axis=0)

    # IFFT
    res = jnp.fft.ifft(summed_spec, axis=0)
    res = jnp.fft.irfft(res, n=neighbors.shape[-1], axis=1)

    # Unpad
    h_orig = flat_psfs.shape[-2]
    psf = res[n_pad : n_pad + h_orig, n_pad : n_pad + h_orig]

    return jnp.maximum(psf, 0.0)
