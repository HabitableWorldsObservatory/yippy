"""Float64 path. Auto-skips unless the process was started with x64 enabled.

Run with:  JAX_ENABLE_X64=1 uv run pytest tests/test_precision_x64.py -q
"""

import jax
import pytest

if not jax.config.jax_enable_x64:
    pytest.skip(
        "run under JAX_ENABLE_X64=1 to exercise the float64 path",
        allow_module_level=True,
    )

from conftest import assert_eqx_arrays_match_active_dtype


def test_datacube_cache_path_is_f64_keyed(coro):
    """The PSF datacube cache filename carries the 'f64' tag under x64."""
    from yippy._yip import _YIPData

    host = _YIPData(coro.yip_path)
    p = host._datacube_cache_path
    assert p.name.startswith(f"psf_datacube_quarter_f64_{coro.grid.shape[0]}px_")
    assert p.suffix == ".npy"


def test_eqx_stored_arrays_are_f64(eqx_coro):
    """Every stored/forward Coronagraph float array is float64 under x64."""
    assert_eqx_arrays_match_active_dtype(eqx_coro)
