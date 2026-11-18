"""Finite-diameter Images retain the backend's diameter dependence and JAX support."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff import MultiBandCoronagraph, ScalarCoronagraph
from optixstuff.apertures import Circular, HalfMaximum
from optixstuff.grids import resample
from optixstuff.performance import performance_curves
from optixstuff.spectral import Band
from synthetic_yip import (
    replace_stellar_tables,
    synthetic_coronagraph,
)

from yippy import open_yip


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


def test_multiband_forwards_diameter_and_time(cached_diameter_coro):
    """Exact record and image dispatch; monochromatic records ignore time/wavelength."""
    c = cached_diameter_coro
    mb = MultiBandCoronagraph([c, eqx.tree_at(lambda m: m.bandpass, c, Band(800, 900))])
    for wl in (500.0, 850.0):
        np.testing.assert_array_equal(
            mb.band(wl).stellar_map(wl, 0.1, grid=c.grid).data,
            c.stellar_map(wl, 0.1, grid=c.grid).data,
        )
        np.testing.assert_array_equal(
            mb.throughput(2.0, wl, time_d=3.0), c.throughput(2.0)
        )


def test_yippy_diameter_jit_and_vmap(cached_diameter_coro):
    """16 float32 eps for the log spline evaluated eagerly, JIT and vmap."""
    c = cached_diameter_coro

    def f(d):
        return c.stellar_map(500.0, d, grid=c.grid).data

    np.testing.assert_array_equal(
        jax.jit(lambda s: c.leakage_fraction(s))(jnp.array(1.0)),
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


@pytest.fixture(
    scope="module",
    params=[
        False,
        True,
        pytest.param(
            "eac1_aavc_2d",
            marks=[
                pytest.mark.network,
                pytest.mark.slow(
                    reason="delivered maps require full native public-YIP grids"
                ),
            ],
        ),
        pytest.param(
            "usort_offaxis_ovc",
            marks=[
                pytest.mark.network,
                pytest.mark.slow(
                    reason="delivered maps require full native public-YIP grids"
                ),
            ],
        ),
    ],
)
def finite_coro(request, tmp_path_factory):
    """Nonuniform finite-star images, optionally in a distinct stellar frame."""
    from dataclasses import replace

    rules = (Circular(0.7), HalfMaximum())
    with jax.enable_x64():
        if isinstance(request.param, str):
            return tuple(
                open_yip(request.param).prepare(rules=rules, leakage_rule=lr)
                for lr in ("aperture_mean", "azimuthal_ring_mean")
            )
        path = tmp_path_factory.mktemp("finite_stellar_maps")
        base = synthetic_coronagraph(path, rules=rules)
        knots = jnp.array([0.0, 0.1, 0.2])
        x, y = base.stellar_grid.coordinates()
        maps = jnp.stack(
            [
                base.stellar_map(None, 0.0).integrated().data
                * jnp.exp(d * (2 + 0.2 * x + 0.1 * y**2))
                for d in knots
            ]
        )
        grid = base.stellar_grid
        if request.param:
            grid = replace(
                grid,
                frame=replace(
                    grid.frame, diameter_m=4.0, reference_wavelength_nm=600.0
                ),
            )
        return tuple(
            replace_stellar_tables(
                open_yip(path).prepare(rules=rules, leakage_rule=lr), grid, knots, maps
            )
            for lr in ("aperture_mean", "azimuthal_ring_mean")
        )


@pytest.mark.parametrize("rule", [Circular(0.7), HalfMaximum()])
@pytest.mark.parametrize("leakage_rule", ["aperture_mean", "azimuthal_ring_mean"])
def test_finite_leakage_record_reference(finite_coro, rule, leakage_rule):
    """Finite stellar leakage agrees with independently remeasured native maps.

    Floating-point: 1e-9 for DFT cancellation in faint PSF tails, or 64
    stored-dtype eps for float32 leaves and log-spline rounding.

    The reference remeasures the interpolated native map, at diameter knots
    and midpoints. Separations include knots, midpoints and both linear tails.
    Distinct stellar frames exercise the transpose of the spatial sampler.
    """
    with jax.enable_x64():
        c = finite_coro[leakage_rule == "azimuthal_ring_mean"]
        record = c.curves(rule, leakage_rule=leakage_rule)
        axis = record.separation_lod
        s = jnp.array(
            [
                axis[0] - 0.1,
                axis[0],
                (axis[0] + axis[1]) / 2,
                axis[len(axis) // 2],
                (axis[-2] + axis[-1]) / 2,
                axis[-1],
                axis[-1] + 0.1,
            ]
        )
        knots = c._stellar_ln_interp.x
        if c.grid.shape == (256, 256):
            low, high = knots[6:8]
            diameters = jnp.array([0.0, low, (low + high) / 2])
        else:
            diameters = jnp.array([0.0, 0.05, 0.1, 0.15, 0.2])
        backend = c._offaxis_backend
        xs, ys = np.asarray(backend.x_offsets), np.asarray(backend.y_offsets)
        iy = int(np.argmin(np.abs(ys)))
        native = {
            float(x): backend.get_psf_by_offset_idx(ix, iy) for ix, x in enumerate(xs)
        }
        positions = [(float(x), float(ys[iy])) for x in record.separation_lod]
        expected = []
        names = (
            "leakage_fraction",
            "leakage_density_lod2",
            "raw_contrast",
            "leakage_fraction_variance",
            "leakage_density_variance_lod2",
        )
        for d in diameters:
            reference = (
                record
                if d == 0
                else performance_curves(
                    native.__getitem__,
                    c.stellar_map(None, d),
                    rule,
                    separations_lod=record.separation_lod,
                    source_positions_lod=positions,
                    leakage_rule=leakage_rule,
                    stellar_diameter=float(d),
                )
            )
            expected.append(jnp.stack([reference.evaluate(name, s) for name in names]))

        def query(d):
            return jnp.stack(
                [
                    getattr(c, name)(
                        s, stellar_diam_lod=d, rule=rule, leakage_rule=leakage_rule
                    )
                    for name in names[:3]
                ]
                + [
                    method(
                        s,
                        stellar_diam_lod=d,
                        rule=rule,
                        leakage_rule=leakage_rule,
                        variance=True,
                    )
                    for method in (c.leakage_fraction, c.leakage_density_lod2)
                ]
            )

        tol = max(1e-9, 64 * np.finfo(record.throughput.dtype).eps)
        np.testing.assert_allclose(
            jax.jit(jax.vmap(query))(diameters), expected, rtol=tol, atol=0
        )
        np.testing.assert_allclose(
            jnp.stack([query(d) for d in diameters]), expected, rtol=tol, atol=0
        )
        np.testing.assert_array_equal(query(0.0), expected[0])
        projected = c.leakage_density_lod2(
            c.separation_map(),
            rule=rule,
            stellar_diam_lod=0.15,
            leakage_rule=leakage_rule,
        )
        np.testing.assert_array_equal(
            c.leakage_density_map(
                [rule], stellar_diam_lod=0.15, leakage_rule=leakage_rule
            )[..., 0],
            projected,
        )
        mb = MultiBandCoronagraph(
            [c, eqx.tree_at(lambda m: m.bandpass, c, Band(750, 850))]
        )
        np.testing.assert_array_equal(
            mb.leakage_fraction(
                s, 800, rule=rule, leakage_rule=leakage_rule, stellar_diam_lod=0.15
            ),
            query(0.15)[0],
        )
        scalar = ScalarCoronagraph(
            grid=c.grid,
            throughput_fraction=0.3,
            aperture_area_lod2=1.0,
            stellar_leakage_density_lod2=1e-9,
        )
        for name in ("leakage_fraction", "leakage_density_lod2", "raw_contrast"):
            with pytest.raises(ValueError, match="finite stellar diameter"):
                getattr(scalar, name)(1.0, stellar_diam_lod=0.1)
