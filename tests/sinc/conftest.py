"""Fixtures for the sinc tests, which run under binary64."""

import gc

import jax
import pytest

from .helpers import delivery, make_prefiltered, make_reader, tone_delivery


@pytest.fixture(scope="module", autouse=True)
def x64_and_cache_clear():
    """Run each sinc test module under binary64, leaving the global setting alone."""
    with jax.enable_x64(True):
        yield
    jax.clear_caches()
    gc.collect()


@pytest.fixture(scope="module")
def full_delivery():
    return delivery()


@pytest.fixture(scope="module")
def full_reader(full_delivery):
    return make_reader(full_delivery)


@pytest.fixture(scope="module")
def quadrant_delivery():
    return delivery(quadrant=True)


@pytest.fixture(scope="module")
def quadrant_reader(quadrant_delivery):
    return make_reader(quadrant_delivery)


@pytest.fixture(scope="session")
def tone_full():
    with jax.enable_x64(True):
        data, truth, norm = tone_delivery()
        reader, receipt = make_prefiltered(data)
    return dict(data=data, truth=truth, norm=norm, reader=reader, receipt=receipt)


@pytest.fixture(scope="session")
def tone_quadrant():
    with jax.enable_x64(True):
        data, truth, norm = tone_delivery(quadrant=True)
        reader, receipt = make_prefiltered(data)
    return dict(data=data, truth=truth, norm=norm, reader=reader, receipt=receipt)
