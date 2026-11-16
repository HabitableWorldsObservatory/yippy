"""Checks for the shared performance record at the JAX consumer boundary."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import requires_available_yip
from hwoutils.grids import Frame
from scipy.interpolate import PchipInterpolator

from yippy import Coronagraph, EqxCoronagraph
from yippy.apertures import Circular
from yippy.performance import PerformanceCurves


def _record(nodes, values):
    """Build a small host record for interpolation and derivative checks."""
    return PerformanceCurves(
        separation_lod=nodes,
        frame=Frame(500.0, 6.0, "circumscribed"),
        rule=Circular(0.7),
        throughput=values,
        core_area_lod2=np.ones_like(nodes),
        leakage_fraction=np.full_like(nodes, 1e-10),
        leakage_density_lod2=np.full_like(nodes, 1e-10),
        raw_contrast=np.full_like(nodes, 1e-9),
        iwa_lod=nodes[0],
        data_extent_lod=nodes[-1],
    )


class TestPerformanceRecord:
    """Compare JAX evaluation with the independent SciPy PCHIP reference."""

    @pytest.fixture()
    def sine_record(self):
        """Smooth positive data on a uniform separation axis."""
        nodes = np.linspace(0, 2 * np.pi, 50)
        values = 0.5 + 0.25 * np.sin(nodes)
        return _record(nodes, values)

    def test_agrees_at_original_knots(self, sine_record):
        """Fails if JAX changes knot values; tolerance is float32 roundoff."""
        actual = sine_record.evaluate(
            "throughput", jnp.asarray(sine_record.separation_lod, dtype=jnp.float32)
        )
        np.testing.assert_allclose(actual, sine_record.throughput, rtol=2e-6)

    def test_agrees_at_midpoints(self, sine_record):
        """Fails on a spline refit; tolerance allows float32 polynomial rounding."""
        nodes = sine_record.separation_lod
        mids = (nodes[:-1] + nodes[1:]) / 2
        expected = PchipInterpolator(nodes, sine_record.throughput)(mids)
        actual = sine_record.evaluate(
            "throughput", jnp.asarray(mids, dtype=jnp.float32)
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6)

    def test_edge_values(self, sine_record):
        """Fails if linear tails use PCHIP endpoint slopes; float32 roundoff bound."""
        nodes, values = sine_record.separation_lod, sine_record.throughput
        queries = np.array([nodes[0] - 0.1, nodes[0], nodes[-1], nodes[-1] + 0.1])
        expected = np.array(
            [
                values[0] - 0.1 * (values[1] - values[0]) / (nodes[1] - nodes[0]),
                values[0],
                values[-1],
                values[-1] + 0.1 * (values[-1] - values[-2]) / (nodes[-1] - nodes[-2]),
            ]
        )
        actual = sine_record.evaluate(
            "throughput", jnp.asarray(queries, dtype=jnp.float32)
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6)

    def test_monotonic_data(self):
        """Fails if coefficients cease to be PCHIP; float32 polynomial bound."""
        nodes = np.linspace(1.0, 10.0, 30)
        values = np.log(nodes)
        record = _record(nodes, values)
        queries = np.linspace(1.5, 9.5, 20)
        expected = PchipInterpolator(nodes, values)(queries)
        actual = record.evaluate("throughput", jnp.asarray(queries, dtype=jnp.float32))
        np.testing.assert_allclose(actual, expected, rtol=2e-6)


@requires_available_yip
class TestEqxCoronagraphInterpolation:
    """Check host-to-JAX conversion on the public package, at float32 precision."""

    @pytest.fixture()
    def test_separations(self, coro):
        """Sample from the inner crossing to the array extent."""
        return np.linspace(
            coro.performance.iwa_lod, coro.performance.data_extent_lod, 20
        )

    def test_throughput(self, coro, eqx_coro, test_separations):
        """Fails if the wrapper evaluates a different throughput table."""
        expected = coro.performance.evaluate("throughput", test_separations)
        actual = eqx_coro.throughput(test_separations)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-8)

    def test_raw_contrast_at_knots(self, coro, eqx_coro):
        """Fails if the contrast table or its scale changes at the JAX boundary."""
        unfloored = eqx.tree_at(
            lambda model: model.contrast_floor,
            eqx_coro,
            None,
            is_leaf=lambda value: value is None,
        )
        knots = coro.performance.separation_lod
        expected = coro.performance.evaluate("raw_contrast", knots)
        actual = unfloored.raw_contrast(knots)
        np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=0)

    def test_occulter_transmission(self, coro, eqx_coro, test_separations):
        """Fails if the occulter profile is read on the performance axis."""
        expected = coro._occultation_record.evaluate(
            "occulter_transmission", test_separations
        )
        actual = eqx_coro.occulter_transmission(test_separations)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-8)

    def test_core_area(self, coro, eqx_coro, test_separations):
        """Fails if the wrapper changes the aperture area or its lod-squared unit."""
        expected = coro.performance.evaluate("core_area_lod2", test_separations)
        actual = eqx_coro.core_area(test_separations)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-8)

    def test_core_mean_intensity(self, coro, eqx_coro, test_separations):
        """Fails if the stellar profile is read on the performance axis."""
        expected = coro._intensity_record.evaluate(
            "core_mean_intensity", test_separations
        )
        actual = eqx_coro.core_mean_intensity(test_separations)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)


def test_synthetic_performance_agrees_in_float64(synthetic_1d):
    """Fails on host/JAX drift, including tails and diameter-dependent profiles.

    The 1e-12 bound allows float64 polynomial and ring-reader roundoff;
    1e-24 absolute tolerance remains far below the synthetic stellar flux.
    """
    with jax.enable_x64():
        coro = Coronagraph(synthetic_1d.path)
        eqx_coro = EqxCoronagraph(yippy_coro=coro)
        grid = np.array([-2.0, -0.1, 0.0, 0.13, 1.37, 3.29, 6.0, 10.0, 15.0])
        for name in [
            "throughput",
            "raw_contrast",
            "core_area",
            "occulter_transmission",
            "core_mean_intensity",
        ]:
            expected = getattr(coro, name)(grid)
            method = getattr(eqx_coro, name)
            np.testing.assert_allclose(method(grid), expected, rtol=1e-12, atol=1e-24)
            np.testing.assert_allclose(
                jax.jit(lambda x, method=method: method(x))(grid),
                expected,
                rtol=1e-12,
                atol=1e-24,
            )
        for diameter in [0.05, 0.1, 0.3]:
            expected = coro.core_mean_intensity(grid, stellar_diam=diameter)
            actual = jax.jit(lambda x, d: eqx_coro.core_mean_intensity(x, d))(
                grid, diameter
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-24)
        derivative = jax.grad(lambda x: eqx_coro.throughput(x))(1.37)
        expected_derivative = PchipInterpolator(
            coro.performance.separation_lod, coro.performance.throughput
        )(1.37, nu=1)
        np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-12)
        np.testing.assert_allclose(
            eqx.filter_jit(lambda model, x: model.throughput(x))(eqx_coro, grid),
            coro.throughput(grid),
            rtol=1e-12,
            atol=1e-24,
        )
        for removed in (
            "IWA",
            "OWA",
            "_throughput_interp",
            "_log_contrast_interp",
            "_core_area_interp",
        ):
            assert not hasattr(eqx_coro, removed)
        with pytest.raises(TypeError):
            EqxCoronagraph(yippy_coro=coro, interp_order=1)


def test_pchip_conversion_uses_original_nodes():
    """Fails on a cubic refit or halved endpoint gradients; float64 roundoff bound."""
    with jax.enable_x64():
        nodes = np.array([0.0, 0.1, 0.3, 0.7, 1.5, 4.0, 8.0])
        values = np.array([0.0, 0.01, 0.03, 0.5, 0.9, 0.95, 1.0])
        host = _record(nodes, values)
        converted = jax.tree_util.tree_map(
            lambda value: (
                jnp.asarray(value) if isinstance(value, np.ndarray) else value
            ),
            host,
        )
        scipy_interp = PchipInterpolator(nodes, values)
        grid = np.linspace(nodes[0], nodes[-1], 101)
        np.testing.assert_allclose(
            converted.evaluate("throughput", jnp.asarray(grid)),
            scipy_interp(grid),
            rtol=1e-12,
            atol=1e-16,
        )
        np.testing.assert_allclose(
            eqx.filter_jit(lambda record, x: record.evaluate("throughput", x))(
                converted, jnp.asarray(grid)
            ),
            scipy_interp(grid),
            rtol=1e-12,
            atol=1e-16,
        )
        endpoints = nodes[[0, -1]]
        derivatives = jax.vmap(jax.grad(lambda x: converted.evaluate("throughput", x)))(
            endpoints
        )
        np.testing.assert_allclose(
            derivatives, scipy_interp(endpoints, nu=1), rtol=1e-12
        )
