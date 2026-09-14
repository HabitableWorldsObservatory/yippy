"""Shared fixtures for yippy tests."""

import os

import numpy as np
import pytest

from yippy import Coronagraph
from yippy._precision import float_dtype
from yippy.datasets import CATALOG, DATA_RELEASE_TAG


def assert_eqx_arrays_match_active_dtype(eqx_coro):
    """Assert every stored/forward float array on an EqxCoronagraph is the active dtype.

    Checks a directly stored array (``sky_trans``) plus forward evaluations that
    exercise the stellar-intensity spline, a performance interpolator, and the 2D
    core-intensity interpolant. The forward-output dtype doubles as the
    no-float32-leak guard. Shared by the float32 and float64 precision suites.

    Args:
        eqx_coro: An ``EqxCoronagraph`` built under the active x64 setting.
    """
    expected = np.dtype(float_dtype())
    assert np.dtype(eqx_coro.sky_trans.dtype) == expected, "sky_trans"
    assert np.dtype(eqx_coro.stellar_intens(0.0).dtype) == expected, "stellar_intens"
    assert np.dtype(eqx_coro.throughput(5.0).dtype) == expected, "throughput"
    if eqx_coro._has_2d_core_intensity:
        out_2d = eqx_coro.core_mean_intensity(5.0, 1e-3)
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
    """Session-scoped EqxCoronagraph built from the real coronagraph."""
    from yippy.eqx_coronagraph import EqxCoronagraph

    return EqxCoronagraph(yippy_coro=coro)


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

    ``Coronagraph`` is the loader and uses scipy interpolants, so its
    performance methods are not traceable. ``EqxCoronagraph`` is the array-only
    view a consumer calls from inside ``jit``, ``vmap`` or ``grad``.
    """
    from yippy.eqx_coronagraph import EqxCoronagraph

    return EqxCoronagraph(yippy_coro=syn_coro_1d)


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
