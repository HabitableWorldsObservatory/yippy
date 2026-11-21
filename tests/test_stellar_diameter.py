"""Finite-diameter Images retain the backend's diameter dependence and JAX support."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from optixstuff import MultiBandCoronagraph
from optixstuff.grids import resample
from optixstuff.spectral import Band


def test_yippy_diameter_forwarding(cached_diameter_coro):
    """Exact forwarding of native stellar Images, including finite diameters."""
    c = cached_diameter_coro
    for d in (0.0, 0.1, 0.2):
        np.testing.assert_array_equal(
            c.stellar_map(500.0, d, grid=c.grid).data,
            resample(c.stellar_map(None, d).integrated(), c.grid).data,
        )
    assert float(c.stellar_map(500.0, 0.2, grid=c.grid).data.sum()) > float(
        c.stellar_map(500.0, 0.0, grid=c.grid).data.sum()
    )


def test_multiband_forwards_diameter(cached_diameter_coro):
    """Exact image dispatch; monochromatic records ignore wavelength."""
    c = cached_diameter_coro
    mb = MultiBandCoronagraph([c, eqx.tree_at(lambda m: m.bandpass, c, Band(800, 900))])
    for wl in (500.0, 850.0):
        np.testing.assert_array_equal(
            mb.coronagraph_at(wl).stellar_map(wl, 0.1, grid=c.grid).data,
            c.stellar_map(wl, 0.1, grid=c.grid).data,
        )
        np.testing.assert_array_equal(
            mb.coronagraph_at(wl).curves().evaluate("throughput", 2.0),
            c.curves().evaluate("throughput", 2.0),
        )


def test_yippy_diameter_jit_and_vmap(cached_diameter_coro):
    """16 float32 eps for the log spline evaluated eagerly, JIT and vmap."""
    c = cached_diameter_coro

    def f(d):
        return c.stellar_map(500.0, d, grid=c.grid).data

    np.testing.assert_array_equal(
        jax.jit(lambda s: c.curves().evaluate("leakage_fraction", s))(jnp.array(1.0)),
        c.curves().evaluate("leakage_fraction", jnp.array(1.0)),
    )
    diameters = jnp.array([0.0, 0.1, 0.2])
    expected = jnp.stack([f(d) for d in diameters])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(f))(diameters),
        expected,
        rtol=16 * np.finfo(np.float32).eps,
        atol=0.0,
    )
