"""Precision policy for yippy.

All float storage follows the global ``jax_enable_x64`` flag: float32 when the
flag is off (the default, memory-friendly), float64 when it is on. This avoids a
segmented float32/float64 pipeline, which JAX handles poorly.
"""

import jax
import numpy as np


def float_dtype():
    """Active default float dtype: float64 if jax_enable_x64 else float32.

    Returns:
        A numpy dtype usable for both ``np.*`` and ``jnp.*`` allocations.
    """
    return jax.dtypes.canonicalize_dtype(np.float64)


def dtype_tag():
    """Short cache key for the active float dtype.

    Returns:
        ``"f64"`` when the active float dtype is 8 bytes wide, else ``"f32"``.
    """
    return "f64" if np.dtype(float_dtype()).itemsize == 8 else "f32"
