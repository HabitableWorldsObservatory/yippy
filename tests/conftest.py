"""Shared fixtures for yippy tests."""

import importlib.util
import os

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import settings
from synthetic_yip import synthetic_coronagraph

from yippy import Coronagraph
from yippy._precision import float_dtype
from yippy.datasets import CATALOG, DATA_RELEASE_TAG
from yippy.performance import stellar_intensity_curves

pytest_plugins = []
# Released hwoutils versions may not include the optional budget plugin yet.
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")
if importlib.util.find_spec("hwoutils.internal_refs") is not None:
    pytest_plugins.append("hwoutils.internal_refs")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))


def assert_eqx_arrays_match_active_dtype(eqx_coro):
    """Assert every stored/forward float array on an Coronagraph is the active dtype.

    Checks a directly stored array (``sky_trans``) plus forward evaluations that
    exercise the stellar-intensity spline, a performance interpolator, and a
    radial intensity record with JAX inputs. The forward-output dtype doubles as the
    no-float32-leak guard. Shared by the float32 and float64 precision suites.

    Args:
        eqx_coro: An ``Coronagraph`` built under the active x64 setting.
    """
    expected = np.dtype(float_dtype())
    assert np.dtype(eqx_coro.sky_trans.data.dtype) == expected, "sky_trans"
    assert (
        np.dtype(eqx_coro.stellar_map(None, 0.0, grid=None).data.dtype) == expected
    ), "stellar_intens"
    assert (
        np.dtype(eqx_coro.curves().evaluate("throughput", jnp.asarray(5.0)).dtype)
        == expected
    ), "throughput"
    if len(eqx_coro._stellar_ln_interp.x) > 1:
        out_2d = stellar_intensity_curves(
            eqx_coro, 1e-3, interpolation="keys"
        ).evaluate("core_mean_intensity", jnp.asarray(5.0))
        assert np.dtype(out_2d.dtype) == expected, "core_mean_intensity_2d"


requires_available_yip = pytest.mark.skipif(
    CATALOG["eac1_aavc_2d"]["md5"] is None or DATA_RELEASE_TAG.endswith("PLACEHOLDER"),
    reason="YIP not yet fetchable: md5 unset or release tag is placeholder.",
)


@pytest.fixture(scope="session")
def coro():
    """Session-scoped real coronagraph loaded from yippy's pooch registry."""
    from yippy import fetch_yip

    yip_path = fetch_yip("eac1_aavc_2d")
    return Coronagraph(yip_path)


@pytest.fixture(scope="session")
def eqx_coro(coro):
    """Session-scoped Coronagraph built from the real coronagraph."""
    return coro


# ---------------------------------------------------------------------------
# Synthetic analytic packages
#
# These need no network and no cache: every quantity they contain has a closed
# form, so a test can compute the right answer from the package constants
# rather than from the reader it is testing. See ``tests/synthetic_yip.py``.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_1d(tmp_path_factory):
    """A radially symmetric analytic package, written once per session."""
    from synthetic_yip import write_synthetic_yip

    return write_synthetic_yip(tmp_path_factory.mktemp("yip1d") / "synthetic_1d")


@pytest.fixture(scope="session")
def synthetic_2d(tmp_path_factory):
    """A quarterly symmetric analytic package, written once per session."""
    from synthetic_yip import write_synthetic_yip

    return write_synthetic_yip(
        tmp_path_factory.mktemp("yip2d") / "synthetic_2d", kind="2dq"
    )


@pytest.fixture(scope="session")
def syn_coro_1d(synthetic_1d):
    """Coronagraph over the radially symmetric analytic package."""
    return Coronagraph(synthetic_1d.path)


@pytest.fixture(scope="session")
def syn_coro_2d(synthetic_2d):
    """Coronagraph over the quarterly symmetric analytic package."""
    return Coronagraph(synthetic_2d.path)


@pytest.fixture(scope="session")
def syn_eqx_1d(syn_coro_1d):
    """The JAX-facing view of the radially symmetric analytic package.

    ``Coronagraph`` builds the host records. ``Coronagraph`` holds their
    JAX leaves for record evaluation and PSF synthesis inside ``jit``,
    ``vmap`` or ``grad``.
    """
    return syn_coro_1d


# A cold cache with no network is a missing referent, not a passing test. The
# default is to skip with a reason that names the cause, so that a contributor
# without the data can still run the physics suite; setting this flag turns
# every such skip into a failure, which is what the nightly and any run that
# claims the evidence should use. A cache that is present and disagrees with
# the contract fails either way.
REFERENCE_DATA_REQUIRED = os.environ.get("YIPPY_REQUIRE_REFERENCE_DATA") == "1"


@pytest.fixture(scope="session")
def shipped_coro():
    """The shipped AAVC package, or a skip that says why it is absent."""
    from yippy import fetch_yip

    try:
        yip_path = fetch_yip("eac1_aavc_2d")
    except Exception as exc:
        message = (
            f"data-absent: eac1_aavc_2d is not cached and could not be fetched ({exc})"
        )
        if REFERENCE_DATA_REQUIRED:
            pytest.fail(message)
        pytest.skip(message)
    return Coronagraph(yip_path)


@pytest.fixture(scope="session")
def cached_diameter_coro(tmp_path_factory):
    """One diameter-dependent analytic table, shared by the contract tests."""
    return synthetic_coronagraph(tmp_path_factory.mktemp("stellar_diameter_yip"))
