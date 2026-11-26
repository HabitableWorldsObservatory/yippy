"""Exact dtype aliases and shared structural checks."""

from dataclasses import fields
from typing import get_args, get_origin

import jax
import jax.numpy as jnp
from hwoutils.precision import PrecisionError
from jaxtyping import Bool, Float, Float32, Int32, UInt32

RealArray = Float[jax.Array, "..."]
FLOAT_DTYPES = (jnp.float32, jnp.float64)
F32 = Float32[jax.Array, "..."]
I32 = Int32[jax.Array, "..."]
U32 = UInt32[jax.Array, "..."]
Boolean = Bool[jax.Array, "..."]


def check_array(value, shape, dtype, *, name="Array"):
    """Reject malformed arrays without inspecting traced values.

    Args:
        value: Array to check.
        shape: Required shape tuple.
        dtype: Iterable of accepted dtypes.
        name: Path of the array for error messages.

    Raises:
        PrecisionError: If a narrower floating dtype is used where only
            float64 is accepted.
        TypeError: If the input is not an array or has another unsupported dtype.
        ValueError: If the array shape differs from the required shape.
    """
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array")
    if value.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {value.shape}")
    expected = tuple(jnp.dtype(d) for d in dtype)
    if value.dtype not in expected:
        if (
            expected == (jnp.float64,)
            and jnp.issubdtype(value.dtype, jnp.floating)
            and value.dtype.itemsize < expected[0].itemsize
        ):
            raise PrecisionError(f"{name} must be float64, got {value.dtype}")
        raise TypeError(f"{name} must have dtype {expected}, got {value.dtype}")


def check_static(module):
    """Keep textual PyTree metadata immutable and correctly typed.

    Args:
        module: Dataclass module whose static fields are checked.
    """
    for field in fields(module):
        if not field.metadata.get("static", False):
            continue
        value = getattr(module, field.name)
        if field.type is str and not isinstance(value, str):
            raise TypeError(f"{field.name} must be a string")
        if get_origin(field.type) is tuple:
            if not isinstance(value, tuple):
                raise TypeError(f"{field.name} must be an immutable tuple")
            if get_args(field.type)[0] is str and any(
                not isinstance(item, str) for item in value
            ):
                raise TypeError(f"{field.name} must contain strings")
