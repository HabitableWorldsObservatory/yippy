"""A source read is differentiable in the source position, including at a node.

The blend reader turns a position into inverse-distance weights and pixel
shifts. A square root of a squared distance has an undefined derivative where
the distance is zero, which is exactly where a position sits on a tabulated
node or at the origin, so a gradient-based fit started there got NaN. The
expected values here are central differences of the same read, so the test
asserts that the derivative exists and is the slope of the value, not any
physics.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from yippy import _jax_funcs as jf
from yippy import open_yip

# Positions in lambda/D on the synthetic packages, whose nodes sit on a 0.5
# grid from 0 to 6: the origin, an exact node, and a generic point. The 1D
# package tabulates only the +x axis, so its node is on the axis.
POSITIONS = {
    "1d": {"origin": (0.0, 0.0), "node": (2.0, 0.0), "generic": (2.3, 1.2)},
    "2dq": {"origin": (0.0, 0.0), "node": (2.0, 1.0), "generic": (2.3, 1.2)},
}
H = 1e-7
# The probe has unit-variance entries, so gradients at ordinary positions are
# order one. An absolute tolerance of 1e-6 is the roundoff of a float64 central
# difference at step H plus its truncation error on the smooth read; the
# origin gradient is itself about 1e-6 because the occulter blocks the core.
ATOL = 1e-6


@pytest.fixture(scope="module")
def backends(synthetic_1d, synthetic_2d):
    """Double-precision readers: a float32 read cannot resolve a 1e-7 step."""
    with jax.enable_x64():
        return {
            "1d": open_yip(synthetic_1d.path).prepare()._offaxis_backend,
            "2dq": open_yip(synthetic_2d.path).prepare()._offaxis_backend,
        }


def _weighted_read(backend):
    """Sum of the source image against a fixed random weighting."""
    probe = {}

    def read(x, y):
        data = backend.create_psfs(jnp.atleast_1d(x), jnp.atleast_1d(y)).data
        if "p" not in probe:
            rng = np.random.default_rng(0)
            probe["p"] = jnp.asarray(rng.standard_normal(data.shape[1:]), data.dtype)
        return jnp.sum(data * probe["p"])

    return read


@pytest.mark.parametrize("name", ["origin", "node", "generic"])
@pytest.mark.parametrize("kind", ["1d", "2dq"])
def test_read_gradient_is_finite_and_matches_finite_difference(kind, name, backends):
    """The derivative of a read is finite and is the slope of the read.

    The 2D blend weights are smooth at a node, so the derivative equals the
    central difference there. The 1D reader interpolates linearly between
    nodes, which leaves a kink at a node: the derivative is the slope on one
    side, so it must equal the forward or the backward difference.
    """
    x, y = POSITIONS[kind][name]
    with jax.enable_x64():
        read = _weighted_read(backends[kind])
        x_, y_ = jnp.float64(x), jnp.float64(y)
        grad = np.array(jax.grad(read, argnums=(0, 1))(x_, y_))
        f0 = read(x_, y_)
        fx = (float(read(x_ + H, y_)), float(read(x_ - H, y_)))
        fy = (float(read(x_, y_ + H)), float(read(x_, y_ - H)))
        f0 = float(f0)
    assert np.all(np.isfinite(grad)), f"gradient is not finite at {name}: {grad}"
    central = np.array([(fx[0] - fx[1]) / (2 * H), (fy[0] - fy[1]) / (2 * H)])
    if kind == "1d" and name == "node":
        forward = np.array([(fx[0] - f0) / H, (fy[0] - f0) / H])
        backward = np.array([(f0 - fx[1]) / H, (f0 - fy[1]) / H])
        assert min(abs(grad[0] - forward[0]), abs(grad[0] - backward[0])) < 1e-5
        assert abs(grad[1] - central[1]) < ATOL
    else:
        assert np.allclose(grad, central, rtol=0.0, atol=ATOL), (grad, central)


def _old_idw_weights(x, y, scale, fx, fy, k=4):
    """The weights before the gradient-safe form, for a value comparison."""
    r_q = jnp.hypot(x, y)
    theta_q = jnp.arctan2(y, x)
    r_n = jnp.hypot(fx, fy)
    theta_n = jnp.arctan2(fy, fx)
    dtheta = (theta_n - theta_q + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    dists = jnp.sqrt((r_n - r_q) ** 2 + (r_q * dtheta) ** 2)
    neg, inds = lax.top_k(-dists, k)
    w = 1.0 / (-neg + 1e-10) ** 2
    return inds, w / jnp.sum(w)


@pytest.mark.parametrize("name", ["origin", "node", "generic"])
def test_idw_weights_equal_the_previous_formula(name, backends):
    """Making the distance gradient-safe changes no weight."""
    backend = backends["2dq"]
    x, y = POSITIONS["2dq"][name]
    with jax.enable_x64():
        fx = jnp.asarray(backend.flat_x_offsets, jnp.float64)
        fy = jnp.asarray(backend.flat_y_offsets, jnp.float64)
        inds, weights, _, _ = jf.idw_read_plan(
            jnp.float64(x), jnp.float64(y), 0.25, fx, fy, True, True, "2dq", 4
        )
        old_inds, old_weights = _old_idw_weights(
            jnp.float64(x), jnp.float64(y), 0.25, fx, fy
        )
    assert np.array_equal(inds, old_inds)
    assert np.allclose(weights, old_weights, rtol=1e-12, atol=0.0)
