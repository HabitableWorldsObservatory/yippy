"""A source is read only inside the hull of its tabulated positions.

The synthetic packages tabulate sources out to 6 lambda/D. The stamp of the
default package is larger than that; the narrow package here has a stamp
half-width of 2 lambda/D, so the nodes between 2 and 6 hold only the measured
wing of their source.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from synthetic_yip import write_synthetic_yip

from yippy import open_yip

HULL = 6.0
KINDS = ("1d", "2dq")
# Central-difference step: small enough that the truncation error on the smooth
# read is far below the tolerance, large enough for float64 roundoff.
H = 1e-7


@pytest.fixture(scope="module")
def coros(tmp_path_factory):
    """Coronagraphs on a stamp narrower than the node hull."""
    root = tmp_path_factory.mktemp("domain")
    out = {
        kind: open_yip(
            write_synthetic_yip(root / kind, kind=kind, npix=16).path
        ).prepare()
        for kind in KINDS
    }
    yield out
    jax.clear_caches()


def _read(coro, x, y):
    wavelength = coro.frame.reference_wavelength_nm
    return coro.source_psfs(wavelength, jnp.atleast_1d(x), jnp.atleast_1d(y))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("angle", [0.0, 0.6])
def test_source_beyond_the_hull_is_flagged_and_zero(coros, kind, angle):
    """Fails if a source at 1.4 times the hull is valid or has nonzero light.

    The hull sources are checked beside it: the outermost node itself stays
    valid and keeps its image.
    """
    r = 1.4 * HULL
    result = _read(
        coros[kind],
        jnp.array([r * np.cos(angle), HULL, 1.0]),
        jnp.array([r * np.sin(angle), 0.0, 0.5]),
    )
    np.testing.assert_array_equal(np.asarray(result.valid), [False, True, True])
    data = np.asarray(result.image.data)
    assert np.isfinite(data).all()
    assert (data[0] == 0.0).all()
    assert data[1].sum() > 0


@pytest.mark.parametrize("kind", KINDS)
def test_nonfinite_source_is_zero_and_flagged(coros, kind):
    """Fails if a NaN coordinate gives NaN light or a True flag."""
    result = _read(coros[kind], jnp.array([jnp.nan, 1.0]), jnp.array([0.0, 0.5]))
    np.testing.assert_array_equal(np.asarray(result.valid), [False, True])
    assert np.isfinite(np.asarray(result.image.data)).all()


def test_one_dimensional_wing_between_stamp_and_hull_is_kept(coros):
    """Fails if the 1D read drops the tabulated wing past the stamp edge.

    The stamp half-width is 2 lambda/D and the last node is at 6, so a source
    at 4 sits off the stamp; its tabulated wing is on it and is nonzero.
    """
    coro = coros["1d"]
    stamp = coro.grid.shape[0] * coro.grid.step / 2
    assert stamp < 4.0 < HULL
    result = _read(coro, jnp.array([4.0, 5.9]), jnp.array([0.0, 0.0]))
    assert np.asarray(result.valid).all()
    assert (np.asarray(result.image.data).sum(axis=(-1, -2)) > 0).all()


def _probe(coro):
    wavelength = coro.frame.reference_wavelength_nm
    rng = np.random.default_rng(1)
    weights = jnp.asarray(rng.standard_normal(coro.grid.shape))

    def summed(x, y):
        read = coro.source_psfs(wavelength, jnp.atleast_1d(x), jnp.atleast_1d(y))
        return jnp.sum(read.image.data[0] * weights)

    return summed


# The 1D read is piecewise linear in radius and the ring read bilinear, so the
# reverse mode is exact away from a node; at a node it is one subgradient. The
# points avoid nodes (nodes lie on a 0.5 grid and rings at 1, 1.6, 2.9) except
# the outermost one, whose derivative is tested against the clamp: beyond the
# hull the value is exactly zero, so the gradient is exactly zero.
POINTS = {
    "inside": (2.3, 1.2),
    "hull_edge": (HULL * 0.999, 0.0),
    "beyond_on_axis": (1.4 * HULL, 0.0),
    "beyond_off_axis": (1.2 * HULL, 1.2 * HULL),
    "beyond_inside_circumradius": (1.15 * HULL, 0.0),
}


@pytest.mark.float64_only(reason="Finite differences at a 1e-7 step")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", list(POINTS))
def test_gradient_matches_central_differences(coros, kind, name):
    """Fails if the source gradient is NaN or not the slope of the read.

    check_grads compares reverse mode with central differences in float64 at
    a point inside, at the hull edge and beyond it. Beyond the hull the read
    is zero, so the differences and the gradient are both zero.
    """
    x, y = (jnp.float64(v) for v in POINTS[name])
    summed = _probe(coros[kind])
    check_grads(summed, (x, y), order=1, modes=("rev",), atol=1e-5, rtol=1e-5, eps=H)
    grad = np.asarray(jax.grad(summed, argnums=(0, 1))(x, y))
    assert np.isfinite(grad).all()
    if name.startswith("beyond"):
        assert (grad == 0.0).all()
