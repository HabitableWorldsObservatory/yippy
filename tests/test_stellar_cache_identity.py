"""The stellar interpolation revision invalidates cached performance records."""

import numpy as np
from synthetic_yip import write_synthetic_yip

import yippy._yip as yip_module
from yippy import open_yip
from yippy._yip import _YIPData


def test_performance_cache_misses_base_stellar_interpolation(tmp_path, monkeypatch):
    """A persisted base record misses and a current record hits."""
    write_synthetic_yip(tmp_path, npix=24, max_offset_lod=2.0)
    package = open_yip(tmp_path)
    current_identity = _YIPData._perf_identity

    def base_identity(self, *args, **kwargs):
        identity = current_identity(self, *args, **kwargs)
        identity.pop("stellar_interpolation_revision", None)
        return identity

    with monkeypatch.context() as patch:
        patch.setattr(_YIPData, "_perf_identity", base_identity)
        base = _YIPData(package)
        base_path = base._perf_dir / base._perf_filename()
        assert base_path.is_file()

    performance_curves = yip_module.performance_curves
    computations = []

    def count_computation(*args, **kwargs):
        computations.append(1)
        return performance_curves(*args, **kwargs)

    monkeypatch.setattr(yip_module, "performance_curves", count_computation)
    current = _YIPData(package)
    assert len(computations) == 1
    current_path = current._perf_dir / current._perf_filename()
    assert current_path.is_file() and current_path != base_path

    loaded = _YIPData(package)
    assert len(computations) == 1
    np.testing.assert_array_equal(
        loaded.performance.leakage_density_lod2,
        current.performance.leakage_density_lod2,
    )
