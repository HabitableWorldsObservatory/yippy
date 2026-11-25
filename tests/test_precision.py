"""Check stored and evaluated dtypes and cache keys in each precision lane."""

import jax.numpy as jnp
import numpy as np

from yippy._yip import _YIPData
from yippy.performance import stellar_intensity_curves


def test_datacube_cache_path_is_dtype_keyed(coro, precision):
    """The datacube cache name records the lane dtype, shape and source digest."""
    host = _YIPData(coro.package)
    p = host._datacube_cache_path
    tag = f"f{precision.itemsize * 8}"
    assert p.name.startswith(f"psf_datacube_quarter_{tag}_{coro.grid.shape[0]}px_")
    assert p.suffix == ".npy"
    assert p.parent == host._cache_dir


def test_eqx_arrays_match_lane_dtype(eqx_coro, precision):
    """Stored arrays and forward evaluations use the lane dtype."""
    assert np.dtype(eqx_coro.sky_trans.data.dtype) == precision, "sky_trans"
    assert (
        np.dtype(eqx_coro.stellar_map(None, 0.0, grid=None).data.dtype) == precision
    ), "stellar_intens"
    assert (
        np.dtype(eqx_coro.curves().evaluate("throughput", jnp.asarray(5.0)).dtype)
        == precision
    ), "throughput"
    if len(eqx_coro._stellar_ln_interp.x) > 1:
        out_2d = stellar_intensity_curves(
            eqx_coro, 1e-3, interpolation="keys"
        ).evaluate(jnp.asarray(5.0))
        assert np.dtype(out_2d.dtype) == precision, "core_mean_intensity_2d"
