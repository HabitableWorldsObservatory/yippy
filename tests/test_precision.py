"""Precision policy: yippy follows the global jax_enable_x64 flag (default float32).

The default test process runs x64-off, so these assert the float32 behavior. The
float64 mirror lives in ``test_precision_x64.py``. Both exercise the ``_precision``
helpers through real yippy code (cache filename + stored Coronagraph arrays),
so a dedicated unit test of the trivial helpers would be redundant.
"""

from conftest import assert_eqx_arrays_match_active_dtype


def test_datacube_cache_path_is_dtype_keyed(coro):
    """The PSF datacube cache filename carries the active dtype tag (f32 default).

    Also carries the realized PSF pixel shape and a source-file signature (see
    ``Coronagraph._source_signature``), so the exact name is not pinned here --
    only the parts this test is actually about.
    """
    from yippy._yip import _YIPData

    host = _YIPData(coro.yip_path)
    p = host._datacube_cache_path
    assert p.name.startswith(f"psf_datacube_quarter_f32_{coro.grid.shape[0]}px_")
    assert p.suffix == ".npy"
    assert p.parent == host._cache_dir


def test_eqx_stored_arrays_are_f32_by_default(eqx_coro):
    """In the default (x64-off) suite, Coronagraph float arrays are float32."""
    assert_eqx_arrays_match_active_dtype(eqx_coro)
