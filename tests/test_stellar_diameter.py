"""Finite-diameter Images retain the backend's diameter dependence and JAX support."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from footprint_truth import assert_stellar_read
from optixstuff import MultiBandCoronagraph
from optixstuff.grids import Measure, coverage
from optixstuff.spectral import Band

from yippy.sampling import STELLAR_RECONSTRUCTION


def test_yippy_diameter_forwarding(cached_diameter_coro):
    """Fails if a gridded stellar map drops the diameter of its native samples.

    Truth: the SciPy footprint read of the native samples at each diameter
    (``footprint_truth``), compared to rounding on the pixels the read
    supports (corners two cells inside the table); the outer two rings depend
    on the spline's end condition and are not a promised value.
    """
    c = cached_diameter_coro
    supported = coverage(
        c.stellar_grid,
        c.grid,
        reconstruction=STELLAR_RECONSTRUCTION,
        measure=Measure.density("grid"),
    )
    for d in (0.0, 0.1, 0.2):
        assert_stellar_read(
            c.stellar_map(500.0, d, grid=c.grid).data,
            c.stellar_map(None, d).integrated().data,
            c.stellar_grid,
            c.grid,
            where=supported,
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
        jax.jit(lambda s: c.curves().evaluate("leakage_fraction", s, 0.0))(
            jnp.array(1.0)
        ),
        c.curves().evaluate("leakage_fraction", jnp.array(1.0), 0.0),
    )
    diameters = jnp.array([0.0, 0.1, 0.2])
    expected = jnp.stack([f(d) for d in diameters])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(f))(diameters),
        expected,
        rtol=16 * np.finfo(np.float32).eps,
        atol=0.0,
    )
