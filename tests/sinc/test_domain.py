"""Signed geometric permission boundaries independent of interpolation coverage."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import yippy.sinc as sc
from yippy.sinc.geometry import spacing_xy


@pytest.mark.parametrize(
    "kind,boundary",
    [
        ("box", "closed"),
        ("box", "lower_closed_upper_open"),
        ("annulus", "closed"),
        ("annulus", "lower_closed_upper_open"),
    ],
)
def test_permission_boundaries(kind, boundary):
    """Fails on expanded bounds, unsigned folding or annulus-box substitution."""
    domain = sc.Domain(
        kind,
        boundary,
        jnp.array([[-2.0, -1.0], [1.0, 2.0]]) if kind == "box" else jnp.zeros((2, 2)),
        jnp.array([1.0, 2.0]) if kind == "annulus" else jnp.zeros(2),
    )
    if kind == "box":
        queries = jnp.array(
            [
                [-2.0, -1.0],
                [1.0, 0.0],
                [math.nextafter(1.0, -math.inf), 0.0],
                [math.nextafter(-2.0, -math.inf), 0.0],
                [0.0, 2.0],
            ]
        )
        expected = [True, boundary == "closed", True, False, boundary == "closed"]
    else:
        queries = jnp.array(
            [
                [1.0, 0.0],
                [-2.0, 0.0],
                [math.nextafter(2.0, math.inf), 0.0],
                [0.0, 0.0],
                [1.5, 1.5],
                [0.0, -1.5],
            ]
        )
        expected = [True, boundary == "closed", False, False, False, True]
    np.testing.assert_array_equal(
        eqx.filter_jit(jax.vmap(lambda d, q: d.contains(q), in_axes=(None, 0)))(
            domain, queries
        ),
        expected,
    )


def test_subnormal_host_rejection(full_delivery):
    """Fails if denormal hardware modes decide metadata admission."""
    with pytest.raises(ValueError):
        sc.Domain("box", "closed", jnp.array([[5e-324, 0.0], [1.0, 1.0]]), jnp.zeros(2))
    grid = full_delivery["declared_lattice"]
    data = dict(
        full_delivery,
        declared_lattice=sc.source_grid(
            [5e-324, 0.0], spacing_xy(grid), grid.shape, grid.frame
        ),
    )
    assert sc.admit_grid(**data).reason_bits & sc.Reason.GRID_NONUNIFORM
