"""Tests for yippy.datasets - catalog integrity and public API."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pooch
import pytest

from yippy import datasets

VALID_HASH = re.compile(r"^md5:[0-9a-f]{32}$")


VALID_SAMPLING_SUFFIXES = ("_1d", "_2d")


def test_catalog_keys_match_telescope_coronagraph_sampling_when_conventional():
    """Conventionally-named keys must equal ``{telescope}_{coronagraph}_(1d|2d)``.

    Entries that omit the suffix are exempt and must instead set
    ``sampling`` explicitly (covered by :func:`test_catalog_sampling_set`).
    """
    for name, meta in datasets.CATALOG.items():
        if not (name.endswith("_1d") or name.endswith("_2d")):
            continue
        prefix = f"{meta['telescope']}_{meta['coronagraph']}"
        assert name.startswith(prefix), f"key {name!r} doesn't start with {prefix!r}"
        suffix = name[len(prefix) :]
        assert suffix in VALID_SAMPLING_SUFFIXES, (
            f"key {name!r} has unknown sampling suffix {suffix!r}"
        )


def test_catalog_required_fields():
    """Every entry must carry coronagraph, designer, md5, and sampling.

    ``telescope`` is optional for design-study YIPs that have no fixed
    telescope-architecture pairing.
    """
    required = {"coronagraph", "designer", "md5", "sampling"}
    for name, meta in datasets.CATALOG.items():
        missing = required - set(meta)
        assert not missing, f"{name!r} missing fields: {missing}"


def test_catalog_md5_format():
    """Every catalog entry must have a valid ``md5:<32 hex>`` hash."""
    for name, meta in datasets.CATALOG.items():
        md5 = meta["md5"]
        assert md5 is not None and VALID_HASH.match(md5), (
            f"{name!r} has bad md5: {md5!r}"
        )


def test_fetch_coronagraph_removed():
    """Hard break of the legacy API."""
    assert not hasattr(datasets, "fetch_coronagraph")


def test_list_yips_no_filter_returns_all():
    """list_yips() with no filters returns all catalog keys."""
    names = datasets.list_yips()
    assert sorted(names) == sorted(datasets.CATALOG.keys())


def test_list_yips_telescope_filter():
    """Filtering by telescope returns only matching entries."""
    names = datasets.list_yips(telescope="eac1")
    assert sorted(names) == ["eac1_aavc_2d", "eac1_optimal_order_6_1d"]


def test_list_yips_coronagraph_filter():
    """Filtering by coronagraph isolates one entry in the minimal catalog."""
    assert datasets.list_yips(coronagraph="aavc") == ["eac1_aavc_2d"]
    assert datasets.list_yips(coronagraph="optimal_order_6") == [
        "eac1_optimal_order_6_1d"
    ]


def test_list_yips_sampling_filter():
    """Filtering by sampling returns all entries of that regime."""
    assert datasets.list_yips(sampling="2d") == ["eac1_aavc_2d"]
    assert sorted(datasets.list_yips(sampling="1d")) == [
        "eac1_optimal_order_6_1d",
        "usort_offaxis_ovc",
    ]


def test_list_yips_three_axis_filter_unique():
    """Telescope + coronagraph + sampling narrows to exactly one entry."""
    assert datasets.list_yips(telescope="eac1", coronagraph="aavc", sampling="2d") == [
        "eac1_aavc_2d"
    ]


def test_list_yips_unknown_kwarg_raises_typeerror():
    """An unknown filter keyword raises TypeError with a helpful message."""
    with pytest.raises(TypeError, match="unknown filter"):
        datasets.list_yips(telescpoe="eac1")  # typo intentional


def test_list_yips_no_match_returns_empty():
    """list_yips() with no matching filters returns an empty list."""
    assert datasets.list_yips(telescope="lmt_42") == []


def test_yip_exists_known_name():
    """yip_exists() returns True for any catalog entry."""
    assert datasets.yip_exists("eac1_aavc_2d") is True


def test_yip_exists_unknown_name():
    """yip_exists() returns False for an unknown YIP name."""
    assert datasets.yip_exists("not_a_real_yip") is False


def test_yip_info_returns_metadata_dict():
    """yip_info() returns the catalog metadata dict for a known entry."""
    info = datasets.yip_info("eac1_aavc_2d")
    assert info["telescope"] == "eac1"
    assert info["coronagraph"] == "aavc"
    assert "md5" in info


def test_yip_info_unknown_raises_keyerror():
    """yip_info() raises KeyError for an unknown YIP name."""
    with pytest.raises(KeyError, match="not_a_real_yip"):
        datasets.yip_info("not_a_real_yip")


def test_fetch_yip_both_name_and_kwargs_raises():
    """Passing both a name and filter kwargs raises TypeError."""
    with pytest.raises(TypeError, match="either"):
        datasets.fetch_yip("eac1_aavc_2d", telescope="eac1")


def test_fetch_yip_neither_name_nor_kwargs_raises():
    """Passing neither a name nor filter kwargs raises TypeError."""
    with pytest.raises(TypeError, match="either"):
        datasets.fetch_yip()


def test_fetch_yip_unknown_name_raises_keyerror_with_suggestions():
    """fetch_yip() suggests close catalog names when the lookup fails."""
    with pytest.raises(KeyError) as exc:
        datasets.fetch_yip("eac1_aavc")  # missing sampling suffix
    # difflib should suggest one of the eac1_aavc_* entries
    assert "eac1_aavc" in str(exc.value)


def test_fetch_yip_zero_match_query_raises_valueerror():
    """A structured query with no matches raises ValueError."""
    with pytest.raises(ValueError, match="No YIP matches"):
        datasets.fetch_yip(telescope="lmt_42", coronagraph="aavc")


def test_fetch_yip_multi_match_query_raises_valueerror():
    """A structured query that matches multiple entries raises ValueError.

    With the minimal two-YIP catalog, ``telescope='eac1'`` matches both
    entries because they share that prefix.
    """
    with pytest.raises(ValueError, match="multiple"):
        datasets.fetch_yip(telescope="eac1")


def test_cache_dir_returns_platform_default_when_env_unset(monkeypatch):
    """cache_dir() returns the platform default when YIPPY_CACHE_DIR is unset."""
    from pathlib import Path

    monkeypatch.delenv(datasets.CACHE_DIR_ENV_VAR, raising=False)
    result = datasets.cache_dir()
    assert isinstance(result, Path)
    assert result == Path(datasets._PIKACHU.path)


def test_cache_dir_honors_env_var(monkeypatch, tmp_path):
    """cache_dir() returns the YIPPY_CACHE_DIR location when the env var is set."""
    monkeypatch.setenv(datasets.CACHE_DIR_ENV_VAR, str(tmp_path))
    assert datasets.cache_dir() == tmp_path


def test_cache_dir_expands_user_in_env_var(monkeypatch):
    """A leading ~ in YIPPY_CACHE_DIR is expanded to the user's home directory."""
    from pathlib import Path

    monkeypatch.setenv(datasets.CACHE_DIR_ENV_VAR, "~/Documents/YIPs")
    assert datasets.cache_dir() == Path("~/Documents/YIPs").expanduser()


def test_fetch_yip_env_var_routes_fetch_to_custom_dir(monkeypatch, tmp_path):
    """YIPPY_CACHE_DIR (without cache_path kwarg) routes fetches to that dir."""
    monkeypatch.setenv(datasets.CACHE_DIR_ENV_VAR, str(tmp_path))
    captured = {}

    def fake_fetch(self, fname, processor=None):
        captured["path"] = self.path
        target = self.path / f"{fname}.unzip" / fname.removesuffix(".zip")
        target.mkdir(parents=True, exist_ok=True)
        sample = target / "offax_psf.fits"
        sample.touch()
        return [str(sample)]

    monkeypatch.setattr(pooch.Pooch, "fetch", fake_fetch)
    datasets.fetch_yip("eac1_aavc_2d")

    assert captured["path"] == tmp_path


def test_fetch_yip_cache_path_overrides_env_var(monkeypatch, tmp_path):
    """An explicit cache_path beats YIPPY_CACHE_DIR."""
    env_dir = tmp_path / "env"
    explicit_dir = tmp_path / "explicit"
    env_dir.mkdir()
    explicit_dir.mkdir()
    monkeypatch.setenv(datasets.CACHE_DIR_ENV_VAR, str(env_dir))
    captured = {}

    def fake_fetch(self, fname, processor=None):
        captured["path"] = self.path
        target = self.path / f"{fname}.unzip" / fname.removesuffix(".zip")
        target.mkdir(parents=True, exist_ok=True)
        sample = target / "offax_psf.fits"
        sample.touch()
        return [str(sample)]

    monkeypatch.setattr(pooch.Pooch, "fetch", fake_fetch)
    datasets.fetch_yip("eac1_aavc_2d", cache_path=explicit_dir)

    assert captured["path"] == explicit_dir


def test_fetch_yip_cache_path_builds_separate_pooch(monkeypatch, tmp_path):
    """Passing cache_path routes the fetch through a fresh pooch at that dir."""
    captured = {}

    def fake_fetch(self, fname, processor=None):
        captured["path"] = self.path
        captured["fname"] = fname
        # Mimic Unzip's output: a single file path inside an unzipped folder.
        target = self.path / f"{fname}.unzip" / fname.removesuffix(".zip")
        target.mkdir(parents=True, exist_ok=True)
        sample = target / "offax_psf.fits"
        sample.touch()
        return [str(sample)]

    monkeypatch.setattr(pooch.Pooch, "fetch", fake_fetch)

    result = datasets.fetch_yip("eac1_aavc_2d", cache_path=tmp_path)

    assert captured["path"] == tmp_path
    assert captured["fname"] == "eac1_aavc_2d.zip"
    assert result.endswith("eac1_aavc_2d")


@pytest.mark.network
@pytest.mark.skipif(
    datasets.DATA_RELEASE_TAG.endswith("PLACEHOLDER"),
    reason="Release tag not yet set; data-vN release pending.",
)
def test_fetch_smallest_yip_loads_as_coronagraph(tmp_path):
    """End-to-end: pull the smallest available YIP and load yippy.Coronagraph."""
    from pathlib import Path

    from yippy import Coronagraph

    yip_path = Path(datasets.fetch_yip("eac1_aavc_2d"))
    for required in (
        "offax_psf.fits",
        "offax_psf_offset_list.fits",
        "stellar_intens.fits",
        "stellar_intens_diam_list.fits",
        "sky_trans.fits",
    ):
        assert (yip_path / required).is_file(), f"missing {required}"

    coro = Coronagraph(yip_path)
    assert coro is not None


def _unpacked(root, name, clutter=()):
    """An unzip folder holding a YIP folder plus clutter, listed in os.walk order."""
    yip = root / name
    yip.mkdir(parents=True)
    for f in ("yip.fits", "offax_psf.fits"):
        (yip / f).write_bytes(b"")
    for f in clutter:
        (root / f).write_bytes(b"")
    return [Path(d) / f for d, _, files in os.walk(root) for f in sorted(files)]


def test_yip_dir_ignores_clutter_in_the_unzip_folder(tmp_path):
    """A file the archive never held, listed first, does not move the YIP folder."""
    root = tmp_path / "eac1_aavc_2d.zip.unzip"
    paths = _unpacked(root, "eac1_aavc_2d", clutter=(".DS_Store", "Thumbs.db"))
    assert paths[0].parent == root  # the clutter is listed first
    assert datasets._yip_dir(paths, "eac1_aavc_2d") == root / "eac1_aavc_2d"


def test_yip_dir_without_clutter(tmp_path):
    """The YIP folder is found when the unzip folder holds only the archive."""
    root = tmp_path / "eac1_aavc_2d.zip.unzip"
    paths = _unpacked(root, "eac1_aavc_2d")
    assert datasets._yip_dir(paths, "eac1_aavc_2d") == root / "eac1_aavc_2d"


def test_fetch_yip_returns_the_yip_folder_despite_clutter(tmp_path, monkeypatch):
    """fetch_yip returns the YIP folder, not the unzip folder, despite clutter."""
    root = tmp_path / "eac1_aavc_2d.zip.unzip"
    paths = _unpacked(root, "eac1_aavc_2d", clutter=(".DS_Store",))

    class Stub:
        path = tmp_path

        def fetch(self, fname, processor=None):
            return [str(p) for p in paths]

    monkeypatch.setattr(datasets, "_PIKACHU", Stub())
    monkeypatch.delenv(datasets.CACHE_DIR_ENV_VAR, raising=False)
    assert datasets.fetch_yip("eac1_aavc_2d") == str(root / "eac1_aavc_2d")
