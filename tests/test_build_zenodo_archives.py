"""Tests for scripts/build_zenodo_archives.py."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from astropy.io import fits

# Import the script as a module. Add scripts/ to sys.path during test
# collection - keeps the script self-contained.
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import build_zenodo_archives as bza  # noqa: E402

CANONICAL_FILES = (
    "offax_psf.fits",
    "offax_psf_offset_list.fits",
    "stellar_intens.fits",
    "stellar_intens_diam_list.fits",
    "sky_trans.fits",
)


def _make_fake_yip(yip_dir: Path) -> None:
    """Populate `yip_dir` with the 5 canonical files and some excluded cruft.

    Writes minimal valid FITS files for the canonical names plus cache
    artefacts (`.npy`, `.DS_Store`, `coro_perf.fits*`, `yippy_cache/`).
    """
    yip_dir.mkdir(parents=True, exist_ok=True)
    for name in CANONICAL_FILES:
        hdu = fits.PrimaryHDU(data=[[0.0]])
        hdu.writeto(yip_dir / name, overwrite=True)
    # Cache cruft (should be excluded)
    (yip_dir / "psf_datacube_quarter.npy").write_bytes(b"cache")
    (yip_dir / ".DS_Store").write_bytes(b"mac")
    (yip_dir / "coro_perf.fits").write_bytes(b"cache_fits")
    (yip_dir / "coro_perf.fits_v1.2.3.dev4+abc.fits").write_bytes(b"v_cache")
    (yip_dir / "yippy_cache").mkdir()
    (yip_dir / "yippy_cache" / "stale.pkl").write_bytes(b"stale")


def test_locate_source_dir_case_insensitive(tmp_path):
    """The script should find EAC1/AAVC for catalog entry eac1_aavc."""
    src = tmp_path / "for_rus"
    _make_fake_yip(src / "EAC1" / "AAVC")
    found = bza.locate_source_dir(src, telescope="eac1", coronagraph="aavc")
    assert found == src / "EAC1" / "AAVC"


def test_locate_source_dir_optimal_maps_correctly(tmp_path):
    """optimal_order_6 should resolve to the Optimal subfolder."""
    src = tmp_path / "for_rus"
    _make_fake_yip(src / "EAC2" / "Optimal")
    found = bza.locate_source_dir(src, telescope="eac2", coronagraph="optimal_order_6")
    assert found == src / "EAC2" / "Optimal"


def test_locate_source_dir_missing_returns_none(tmp_path):
    """A missing source subfolder returns None."""
    src = tmp_path / "for_rus"
    (src / "EAC1").mkdir(parents=True)  # exists but no AAVC subfolder
    assert bza.locate_source_dir(src, telescope="eac1", coronagraph="aavc") is None


def test_locate_source_dir_empty_returns_none(tmp_path):
    """Empty source dir (no canonical files) is treated as missing."""
    src = tmp_path / "for_rus"
    (src / "EAC1" / "APLC").mkdir(parents=True)  # exists but empty
    assert bza.locate_source_dir(src, telescope="eac1", coronagraph="aplc") is None


def test_is_excluded_patterns():
    """Cache-pattern basenames are excluded; canonical filenames are not."""
    excluded = (
        "psf_datacube_quarter.npy",
        ".DS_Store",
        "coro_perf.fits",
        "coro_perf.fits_v1.2.3.fits",
        ".hidden",
    )
    for name in excluded:
        assert bza.is_excluded(name), f"{name!r} should be excluded"
    for name in CANONICAL_FILES:
        assert not bza.is_excluded(name), f"{name!r} should be included"


def test_stage_files_excludes_cache(tmp_path):
    """stage_yip copies exactly the canonical files into staging_root/name."""
    src = tmp_path / "src" / "EAC1" / "AAVC"
    _make_fake_yip(src)
    staged = tmp_path / "staged"
    bza.stage_yip(src, staged, name="eac1_aavc_1d")
    files = sorted(p.name for p in (staged / "eac1_aavc_1d").iterdir())
    assert files == sorted(CANONICAL_FILES)


def test_build_archive_zip_contains_only_canonical(tmp_path):
    """build_archive produces a zip whose members are exactly the canonical files."""
    src = tmp_path / "src" / "EAC1" / "AAVC"
    _make_fake_yip(src)
    out = tmp_path / "dist"
    zip_path, _ = bza.build_archive(src, out, name="eac1_aavc_1d")
    assert zip_path == out / "eac1_aavc_1d.zip"
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
    expected = sorted(f"eac1_aavc_1d/{f}" for f in CANONICAL_FILES)
    assert names == expected


def test_build_archive_md5_is_hex32(tmp_path):
    """build_archive returns a 'md5:' + 32-hex-char string."""
    src = tmp_path / "src" / "EAC1" / "AAVC"
    _make_fake_yip(src)
    _, md5 = bza.build_archive(src, tmp_path / "dist", name="eac1_aavc_1d")
    assert md5.startswith("md5:")
    assert len(md5) == len("md5:") + 32
    assert all(c in "0123456789abcdef" for c in md5[4:])


def test_build_archive_is_deterministic(tmp_path):
    """Running twice produces the same md5 (same file timestamps inside zip)."""
    src = tmp_path / "src" / "EAC1" / "AAVC"
    _make_fake_yip(src)
    _, md5a = bza.build_archive(src, tmp_path / "dist1", name="eac1_aavc_1d")
    _, md5b = bza.build_archive(src, tmp_path / "dist2", name="eac1_aavc_1d")
    assert md5a == md5b


def test_emit_catalog_block_preserves_unbuilt_archives():
    """A rebuilt asset changes its digest and retains discovery metadata."""
    from yippy.datasets import Archive, list_yips, yip_info

    digest = "md5:" + "a" * 32
    block = bza.format_catalog_block({"eac1_aavc_2d": digest})
    archives = eval(f"[{block}]", {"Archive": Archive})
    parsed = {a.name: a for a in archives}
    assert parsed["eac1_aavc_2d"].digest == digest
    assert parsed["eac1_aavc_2d"].designer == "Susan Redmond"
    assert parsed["eac1_optimal_order_6_1d"] == yip_info("eac1_optimal_order_6_1d")
    assert set(parsed) == set(list_yips())
