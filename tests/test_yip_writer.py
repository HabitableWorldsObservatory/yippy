"""Publishing a package must leave its prior generation usable after errors."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from optixstuff.grids import Frame, Image, Measure, PixelGrid

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


def test_point_samples_and_cds_default(tmp_path):
    """Fails if a point sample loses its cell area or its FITS sampling convention."""
    from yippy.offax import OffAx
    from yippy.schema import PRODUCT_FILES

    grid = PixelGrid.focal(4, 0.25, Frame(500.0, 6.0, "circumscribed"))
    values = np.ones((2, 4, 4)) * 3.0
    density = Image(values, grid, ("source",), Measure.density("grid"), "fraction")
    products = dict(
        grid=grid,
        offaxis_psfs=density,
        offsets_xy_lod=[0.0, 1.0],
        stellar_intensity=density,
        stellar_diameters_lod=[0.0, 0.1],
        sky_transmission=np.ones(grid.shape),
    )
    with pytest.raises(ValueError, match="SAMPKIND conflicts"):
        write_yip(
            tmp_path / "contradictory",
            {**products, "sample_kind": "point", "headers": {"SAMPKIND": "integral"}},
        )
    path = write_yip(tmp_path / "encoded", products)
    for name in ("offaxis_psfs", "stellar_intensity"):
        assert fits.getheader(path / PRODUCT_FILES[name])["SAMPKIND"] == "point"
        np.testing.assert_array_equal(
            fits.getdata(path / PRODUCT_FILES[name]), values * 0.25**2
        )
        fits.delval(path / PRODUCT_FILES[name], "SAMPKIND")
    package = open_yip(path)
    kind = package.header.offaxis_psfs.sample_kind
    assert kind == package.header.stellar_intensity.sample_kind == "point"
    offaxis = OffAx(
        package.offaxis_psfs, package.offsets_xy_lod, grid, True, True, sample_kind=kind
    )
    knot = offaxis.get_psf_by_offset_idx(1, 0)
    assert knot.measure == density.measure
    np.testing.assert_array_equal(knot.data, values[1])
    copy = open_yip(write_yip(tmp_path / "explicit", package))
    for name in ("offaxis_psfs", "stellar_intensity"):
        assert copy.fits_header(name)["SAMPKIND"] == "point"
        np.testing.assert_array_equal(getattr(copy, name), getattr(package, name))
