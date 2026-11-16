"""Sampling warnings describe aliasing risk once per Coronagraph instance."""

import warnings

import numpy as np
import pytest
from lod_unit import lod
from synthetic_yip import write_synthetic_yip

from yippy import Coronagraph, HalfMaximum
from yippy.performance import performance_curves


@pytest.mark.parametrize("pixel_lod", [0.5, 0.6])
def test_nyquist_warning_once_per_coronagraph(tmp_path, pixel_lod):
    """Coarse images warn once, including on a cache hit; Nyquist images do not."""
    package = write_synthetic_yip(tmp_path / "sampling", pixscale=pixel_lod)
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            coro = Coronagraph(package.path)
            record = performance_curves(coro, coro.stellar_intens(0.0 * lod), coro.rule)
            throughput = record.throughput
            coro.set_aperture_rule(HalfMaximum(0.5))
        sampling = [w for w in caught if "Nyquist" in str(w.message)]
        assert len(sampling) == (1 if pixel_lod > 0.5 else 0)
        assert np.all(np.isfinite(throughput))
        if sampling:
            assert sampling[0].category is UserWarning
            assert "0.6" in str(sampling[0].message)
            assert "0.5 lambda/D" in str(sampling[0].message)
