"""Tests for the PSF datacube cache: atomic writes and truncated-file recovery."""

from __future__ import annotations

import numpy as np

from yippy.coronagraph import _load_cached_datacube, _save_datacube


def test_save_then_load_round_trips(tmp_path):
    """A saved cube loads back unchanged and leaves no temporary file."""
    path = tmp_path / "psf_datacube.npy"
    cube = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
    _save_datacube(path, cube)
    np.testing.assert_array_equal(_load_cached_datacube(path), cube)
    assert [p.name for p in tmp_path.iterdir()] == ["psf_datacube.npy"]


def test_missing_cache_loads_as_none(tmp_path):
    """No file means no cached cube."""
    assert _load_cached_datacube(tmp_path / "absent.npy") is None


def test_truncated_cache_is_discarded(tmp_path):
    """A cache cut short by an interrupted write is removed so it is rebuilt."""
    path = tmp_path / "psf_datacube.npy"
    _save_datacube(path, np.ones((4, 4, 8, 8)))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    assert _load_cached_datacube(path) is None
    assert not path.exists()
