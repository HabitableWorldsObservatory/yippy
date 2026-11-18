"""Publishing a package must leave its prior generation usable after errors."""

from pathlib import Path

import numpy as np
import pytest
from hwoutils.grids import Frame, PixelGrid

from yippy import open_yip, write_yip


def test_invalid_or_interrupted_write_retains_prior_generation(tmp_path, monkeypatch):
    """A bad offset count and a late I/O failure both preserve all five products."""
    grid = PixelGrid.focal(4, 0.25, Frame(500.0, 6.0, "circumscribed"))
    cube = np.arange(32, dtype=float).reshape(2, 4, 4)
    arrays = dict(
        grid=grid,
        offaxis_psfs=cube,
        offsets_xy_lod=[0.0, 2.0],
        stellar_intensity=cube,
        stellar_diameters_lod=[0.0, 0.1],
        sky_transmission=cube[0],
    )
    directory = write_yip(tmp_path / "native", arrays)
    original = open_yip(directory)
    source = original.source
    with pytest.raises(ValueError, match="offset"):
        write_yip(directory, {**arrays, "offsets_xy_lod": [1.0]})
    assert open_yip(directory).source == source

    replace = Path.replace

    failed = False

    def fail_once(path, dest):
        nonlocal failed
        if dest == directory / "sky_trans.fits" and not failed:
            failed = True
            raise OSError("interrupted write")
        return replace(path, dest)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_once)
        with pytest.raises(OSError, match="interrupted"):
            write_yip(directory, {**arrays, "offaxis_psfs": cube * 2})
    assert open_yip(directory).source == source
    np.testing.assert_array_equal(original.offaxis_psfs, cube)
    write_yip(directory, {**arrays, "offaxis_psfs": cube * 2})
    np.testing.assert_array_equal(open_yip(directory).offaxis_psfs, cube * 2)
