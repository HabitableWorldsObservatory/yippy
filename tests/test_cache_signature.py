"""Source identity and lazy samples survive round trips and model preparation."""

import builtins
import io
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from optixstuff.apertures import Circular

from yippy import open_yip, write_yip
from yippy._yip import _YIPData
from yippy.schema import PRODUCT_FILES


@pytest.mark.parametrize("shipped", [False, True], ids=["synthetic", "public-eac1"])
def test_round_trip_native_products(synthetic_2d, shipped, tmp_path):
    """Fails if a round trip changes sample order, units, cards or corner counts."""
    package = open_yip("eac1_aavc_2d" if shipped else synthetic_2d.path)
    other = open_yip(write_yip(tmp_path / "copy", package))
    assert other.header == package.header
    assert (other.source.digest == package.source.digest) == (
        "SAMPKIND" in package.fits_header("offaxis_psfs")
        and "SAMPKIND" in package.fits_header("stellar_intensity")
    )
    assert other.source.path != package.source.path
    for product, filename in PRODUCT_FILES.items():
        np.testing.assert_array_equal(
            getattr(other, product), getattr(package, product)
        )
        original = package.source.path / filename
        restored = other.source.path / filename
        saved, loaded = fits.getheader(original), fits.getheader(restored)
        if product in ("offaxis_psfs", "stellar_intensity") and "SAMPKIND" not in saved:
            assert loaded.pop("SAMPKIND") == "point"
        assert loaded.tostring() == saved.tostring()
        np.testing.assert_array_equal(fits.getdata(restored), fits.getdata(original))
    for name in ("offaxis_psfs", "stellar_intensity", "sky_transmission"):
        assert (
            getattr(other.header, name).pixel_grid()
            == getattr(package.header, name).pixel_grid()
        )


def test_generation_checks_content(synthetic_1d, tmp_path):
    """Fails if equal size and mtime hide a changed generation before a lazy read."""
    package = open_yip(write_yip(tmp_path / "copy", open_yip(synthetic_1d.path)))
    path = package.source.path / PRODUCT_FILES["sky_transmission"]
    stat = path.stat()
    with fits.open(path, mode="update") as hdus:
        hdus[0].data[0, 0] += 1
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert open_yip(package.source.path).source != package.source
    with pytest.raises(ValueError, match="source files changed"):
        _ = package.sky_transmission


def test_prepare_reuses_native_samples(tmp_path, monkeypatch):
    """Fails if rebuilding a rule or cube reads a source product again."""
    from synthetic_yip import write_synthetic_yip

    fixture = write_synthetic_yip(tmp_path / "tiny", npix=8, max_offset_lod=1.5)
    reads = []
    original = fits.getdata

    def record(path, *args, **kwargs):
        reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(fits, "getdata", record)
    package = open_yip(fixture.path)
    assert not reads
    (fixture.path / PRODUCT_FILES["sky_transmission"]).touch()
    model = package.prepare()
    assert len(reads) == len(PRODUCT_FILES)
    np.testing.assert_array_equal(package.offsets_xy_lod, fixture.offsets)
    assert not package.offaxis_psfs.flags.writeable
    assert not package.offsets_xy_lod.flags.writeable

    def forbid_source_reads(open_file):
        def checked(path, *args, **kwargs):
            if (
                isinstance(path, (str, os.PathLike))
                and Path(path).parent == fixture.path
            ):
                raise AssertionError(f"Reopened native product: {path}")
            return open_file(path, *args, **kwargs)

        return checked

    monkeypatch.setattr(builtins, "open", forbid_source_reads(builtins.open))
    monkeypatch.setattr(io, "open", forbid_source_reads(io.open))
    changed = model.with_rules((Circular(0.65),)).with_scene_operator()
    assert changed.package is package
    assert changed.operators[0].shape == (8, 8)


def test_perf_filename_and_datacube_path_carry_the_signature(synthetic_1d):
    """Host cache filenames embed the current source digest."""
    host = _YIPData(open_yip(synthetic_1d.path))
    sig = host.package.source.digest
    assert sig in host._perf_filename()
    assert sig in host._datacube_cache_path.name
