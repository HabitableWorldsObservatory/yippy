"""Source identities invalidate IO caches when top-level YIP files change."""

from types import SimpleNamespace

from yippy._yip import _YIPData


def _signature(yip_path):
    return _YIPData._source_signature(SimpleNamespace(yip_path=yip_path))


def test_signature_stable_when_files_unchanged(tmp_path):
    """Calling the signature twice with no changes returns the same value."""
    (tmp_path / "offax_psf.fits").write_bytes(b"AAAA")
    assert _signature(tmp_path) == _signature(tmp_path)


def test_signature_changes_on_in_place_overwrite(tmp_path):
    """Overwriting a source FITS file with new content changes the signature.

    This is the exact failure mode flagged in review: same filenames, same
    aperture settings, same yippy version, but different underlying data.
    """
    fits_path = tmp_path / "offax_psf.fits"
    fits_path.write_bytes(b"AAAA")
    before = _signature(tmp_path)

    fits_path.write_bytes(b"BBBBBBBB")  # different size -> different signature
    after = _signature(tmp_path)

    assert before != after


def test_signature_changes_when_a_source_file_is_added(tmp_path):
    """Adding a new top-level FITS file changes the signature."""
    (tmp_path / "offax_psf.fits").write_bytes(b"AAAA")
    before = _signature(tmp_path)

    (tmp_path / "sky_trans.fits").write_bytes(b"CCCC")
    after = _signature(tmp_path)

    assert before != after


def test_signature_ignores_yippy_cache_subdirectory(tmp_path):
    """Files inside yippy_cache/ (the cache's own output) do not feed the signature.

    ``_source_signature`` globs ``yip_path`` non-recursively, so cache
    artifacts it writes itself can never perturb the key that names them.
    """
    (tmp_path / "offax_psf.fits").write_bytes(b"AAAA")
    before = _signature(tmp_path)

    cache_dir = tmp_path / "yippy_cache"
    cache_dir.mkdir()
    (cache_dir / "psf_datacube_quarter_f32_64px_deadbeef.npy").write_bytes(b"cache")

    assert _signature(tmp_path) == before


def test_perf_filename_and_datacube_path_carry_the_signature(synthetic_1d):
    """Host cache filenames embed the current source signature."""
    host = _YIPData(synthetic_1d.path)
    sig = host._source_signature()
    assert sig in host._perf_filename()
    assert sig in host._datacube_cache_path.name
