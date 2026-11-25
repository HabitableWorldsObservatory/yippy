"""Signed geometric permission boundaries independent of interpolation coverage."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import yippy.sinc as sc
from yippy.sinc.evidence import Reason
from yippy.sinc.geometry import source_grid, spacing_xy


@pytest.mark.parametrize(
    "kind,boundary",
    [
        ("box", "closed"),
        ("box", "lower_closed_upper_open"),
        ("annulus", "closed"),
        ("annulus", "lower_closed_upper_open"),
    ],
)
def test_permission_boundaries(kind, boundary, precision):
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
                [np.nextafter(1.0, -np.inf, dtype=precision), 0.0],
                [np.nextafter(-2.0, -np.inf, dtype=precision), 0.0],
                [0.0, 2.0],
            ]
        )
        expected = [True, boundary == "closed", True, False, boundary == "closed"]
    else:
        queries = jnp.array(
            [
                [1.0, 0.0],
                [-2.0, 0.0],
                [np.nextafter(2.0, np.inf, dtype=precision), 0.0],
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


def test_subnormal_host_rejection(full_delivery, precision):
    """Fails if denormal hardware modes decide metadata admission."""
    subnormal = np.nextafter(0.0, 1.0, dtype=precision)
    with pytest.raises(ValueError):
        sc.Domain(
            "box", "closed", jnp.array([[subnormal, 0.0], [1.0, 1.0]]), jnp.zeros(2)
        )
    grid = full_delivery["declared_lattice"]
    data = dict(
        full_delivery,
        declared_lattice=source_grid(
            [subnormal, 0.0], spacing_xy(grid), grid.shape, grid.frame
        ),
    )
    assert sc.admit_grid(**data).reason_bits & Reason.GRID_NONUNIFORM
