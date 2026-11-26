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
                [np.nextafter(1.0, -np.inf, dtype=np.float32), 0.0],
                [np.nextafter(-2.0, -np.inf, dtype=np.float32), 0.0],
                [0.0, 2.0],
            ]
        )
        expected = [True, boundary == "closed", True, False, boundary == "closed"]
    else:
        queries = jnp.array(
            [
                [1.0, 0.0],
                [-2.0, 0.0],
                [np.nextafter(2.0, np.inf, dtype=np.float32), 0.0],
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
    subnormal = np.nextafter(0.0, 1.0, dtype=np.float64)
    with pytest.raises(ValueError):
        sc.Domain(
            "box", "closed", np.array([[subnormal, 0.0], [1.0, 1.0]]), jnp.zeros(2)
        )
    grid = full_delivery["declared_lattice"]
    data = dict(
        full_delivery,
        declared_lattice=source_grid(
            [subnormal, 0.0], spacing_xy(grid), grid.shape, grid.frame
        ),
    )
    assert sc.admit_grid(**data).reason_bits & Reason.GRID_NONUNIFORM


def test_annular_boundary_uses_the_represented_position():
    """Resolve an outer radius between rounded and exact squared distances."""
    point = np.array([1.0, 2.0**-13], dtype=np.float32)
    exact = np.sum(point.astype(np.float64) ** 2)
    domain = sc.Domain(
        "annulus", "closed", np.zeros((2, 2)), [0.0, np.sqrt(exact - 2.0**-28)]
    )
    assert not bool(eqx.filter_jit(domain.contains)(jnp.asarray(point)))


@pytest.mark.parametrize("boundary", ["closed", "lower_closed_upper_open"])
def test_annular_sum_rounds_to_binary64(boundary, precision):
    """A squared minor axis below half a binary64 ulp ties at the outer edge."""
    domain = sc.Domain("annulus", boundary, np.zeros((2, 2)), [0.0, 1.0])
    points = np.array([[1, 2.0**-30], [1, 2.0**-26], [1, 0]], dtype=precision)
    expected = [boundary == "closed", False, boundary == "closed"]
    np.testing.assert_array_equal(domain._contains_host(points), expected)
    for contains in (domain.contains, eqx.filter_jit(domain.contains)):
        assert [bool(contains(jnp.asarray(q))) for q in points] == expected


@pytest.mark.parametrize("kind", ["annulus", "box"])
def test_runtime_domain_uses_shared_coordinate_grid(kind, precision):
    """The host resolves a point just outside the closed traced boundary."""
    domain = sc.Domain(
        kind,
        "closed",
        [[-1, -1], [1, 1]] if kind == "box" else np.zeros((2, 2)),
        [0, 1] if kind == "annulus" else [0, 0],
    )
    point = np.array([np.nextafter(1.0, np.inf), 0.0])
    assert not domain.contains(point)
    for runtime in (domain.contains, eqx.filter_jit(domain.contains)):
        assert runtime(jnp.asarray(point, dtype=precision))
