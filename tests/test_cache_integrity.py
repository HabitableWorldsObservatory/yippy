"""Cached results must not depend on the history of earlier calculations.

Each test compares an object built after some earlier call sequence (warm)
against an object built from an identical package written to a separate
directory, which cannot share cache entries with it (cold). The two must agree
to the precision of the stored table, and a cache entry written under different
settings must never be reused.

Performance tables preserve float64 values through FITS storage. Independent
calculations still start with the active JAX dtype, which defaults to float32;
``RTOL = 1e-6`` allows kernel roundoff across cold calculations while staying
below the physical differences guarded here (1e-3 to 1e-1). The storage
round-trip itself is checked for exact equality.
"""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import optixstuff.performance as perf
import pytest
from numpy.testing import assert_allclose
from optixstuff.apertures import Circular, HalfMaximum
from optixstuff.grids import Frame
from synthetic_yip import write_synthetic_yip
from test_band_limited_apertures import reference_circle_weights

import yippy._yip as yip_module
from yippy import open_yip
from yippy._util import (
    load_coro_performance_from_fits,
    read_performance_cache_key,
    save_coro_performance_to_fits,
)
from yippy._yip import _YIPData

RTOL = 1e-6
SEPS = np.array([1.5, 3.0, 4.25])


def test_circular_centres_have_distinct_complete_cache_records(
    synthetic_1d, tmp_path, monkeypatch
):
    """Fails if centre is lost, cache entries alias, or old schemas are reused.

    Basis: exact storage round-trip of all float64 arrays and scalar fields,
    including IWA; a warm load is forbidden from recomputing either estimand.
    """
    from astropy.io import fits

    obj = _YIPData(open_yip(synthetic_1d.path))
    records, paths = [], []
    for centre in ("source", "peak"):
        obj._prepare_record(Circular(0.7, centre=centre))
        record = obj.performance
        records.append(record)
        path = obj._perf_dir / obj._perf_filename()
        paths.append(path)
        header = fits.getheader(path)
        assert header["PERFVER"] == 7
        assert json.loads(header["RULEJSON"])["centre"] == centre
        identity = json.loads(header["CACHEID"])
        assert identity["rule"]["centre"] == centre
        assert float.fromhex(identity["performance_record"]) == 7
        assert record.rule.centre == centre
    assert paths[0] != paths[1]
    assert all(p.exists() for p in paths)
    _forbid_persisted_computation(monkeypatch)
    for original in records:
        loaded = _YIPData(open_yip(synthetic_1d.path), rule=original.rule).performance
        assert loaded.rule == original.rule
        for name in (
            "separation_lod",
            "throughput",
            "throughput_variance",
            "core_area_lod2",
            "core_area_variance_lod2",
            "leakage_fraction",
            "leakage_fraction_variance",
            "leakage_density_lod2",
            "leakage_density_variance_lod2",
            "raw_contrast",
        ):
            np.testing.assert_array_equal(
                getattr(loaded, name), getattr(original, name)
            )
        for name in (
            "iwa_lod",
            "data_extent_lod",
            "frame",
            "leakage_rule",
            "stellar_diameter_lod",
        ):
            assert getattr(loaded, name) == getattr(original, name)
    legacy = tmp_path / "legacy.fits"
    with fits.open(paths[0]) as hdus:
        hdus[0].header["PERFVER"] = 6
        hdus.writeto(legacy)
    assert read_performance_cache_key(legacy) is None
    with pytest.raises(ValueError, match="PERFVER=7"):
        load_coro_performance_from_fits(legacy.name, legacy.parent)


def _yip(tmp_path, name, **kwargs):
    return write_synthetic_yip(tmp_path / name, **kwargs).path


def _metrics(coro):
    return {
        "throughput": np.asarray(coro.performance.evaluate("throughput", SEPS)),
        "core_area": np.asarray(coro.performance.evaluate("core_area_lod2", SEPS)),
        "raw_contrast": np.asarray(coro.performance.evaluate("raw_contrast", SEPS)),
    }


def _assert_same_metrics(warm, cold, names=("throughput", "core_area", "raw_contrast")):
    got, want = _metrics(warm), _metrics(cold)
    for name in names:
        assert_allclose(got[name], want[name], rtol=RTOL, err_msg=name)


def _patch_everywhere(monkeypatch, name, replacement):
    """Replace a performance function in every module that imported it."""
    for module in (perf, yip_module):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, replacement)


def _forbid_persisted_computation(monkeypatch):
    """Make any recomputation of a persisted quantity raise."""

    def fail(*args, **kwargs):
        raise AssertionError("persisted quantity was recomputed")

    for name in ("performance_curves",):
        _patch_everywhere(monkeypatch, name, fail)


def _bump_mtime(directory):
    """Move every source file's mtime forward so the change is unambiguous."""
    for path in Path(directory).glob("*.fits"):
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


# ---------------------------------------------------------------------------
# Prepared truncation records and construction
# ---------------------------------------------------------------------------


def test_prepared_record_then_constructor_matches_cold(tmp_path):
    """Prepared truncation caches reproduce independently constructed records."""
    path = _yip(tmp_path, "warm")
    seed = _YIPData(open_yip(path))
    seed._prepare_record(HalfMaximum(0.3))
    assert seed.rule == HalfMaximum(0.3)
    assert seed.performance is seed.curves_by_rule[HalfMaximum(0.3)]
    warm = _YIPData(open_yip(path), rule=HalfMaximum(0.3))
    cold = _YIPData(open_yip(_yip(tmp_path, "cold")), rule=HalfMaximum(0.3))
    _assert_same_metrics(warm, cold)
    # The record preparation object itself must also match
    _assert_same_metrics(seed, cold)


def test_constructor_then_prepared_record_matches_cold(tmp_path):
    """Preparing truncation after a cache hit matches a cold construction."""
    path = _yip(tmp_path, "warm")
    _YIPData(open_yip(path), rule=HalfMaximum(0.3))
    warm = _YIPData(open_yip(path))
    warm._prepare_record(HalfMaximum(0.3))
    cold = _YIPData(open_yip(_yip(tmp_path, "cold")), rule=HalfMaximum(0.3))
    _assert_same_metrics(warm, cold)


def test_truncation_core_is_read_from_the_cache(tmp_path, monkeypatch, caplog):
    """A cache hit in truncation mode recomputes nothing and warns of nothing."""
    path = _yip(tmp_path, "warm")
    cold = _YIPData(open_yip(path), rule=HalfMaximum(0.3))
    _forbid_persisted_computation(monkeypatch)
    with caplog.at_level("WARNING", logger="yippy"):
        warm = _YIPData(open_yip(path), rule=HalfMaximum(0.3))
    assert not any("not checked" in r.getMessage() for r in caplog.records)
    _assert_same_metrics(warm, cold)


def test_explicit_import_checks_rule_and_leakage(tmp_path):
    """Fails if an explicit import accepts different aperture or leakage metadata."""
    coro = _YIPData(open_yip(_yip(tmp_path, "warm")), rule=HalfMaximum(0.3))
    filename = coro._perf_filename()
    loaded = coro._prepare_record(
        rule=HalfMaximum(0.3),
        save_to_fits=False,
        load_from_file=filename,
        cache_dir=coro._perf_dir,
    )
    np.testing.assert_array_equal(loaded.throughput, coro.performance.throughput)
    for kwargs in (
        {"rule": HalfMaximum(0.5)},
        {"rule": HalfMaximum(0.3), "leakage_rule": "azimuthal_ring_mean"},
    ):
        with pytest.raises(ValueError, match=r"rule|leakage"):
            coro._prepare_record(
                **kwargs,
                save_to_fits=False,
                load_from_file=filename,
                cache_dir=coro._perf_dir,
            )


def test_explicit_floor_does_not_change_cached_contrast(tmp_path):
    """The record preparation stores unfloored contrast, as the constructor does."""
    path = _yip(tmp_path, "warm")
    contrast_floor = 1e-6
    seed = _YIPData(open_yip(path))
    seed._prepare_record(HalfMaximum(0.3))
    warm = _YIPData(open_yip(path), rule=HalfMaximum(0.3))
    cold = _YIPData(open_yip(_yip(tmp_path, "cold")), rule=HalfMaximum(0.3))
    assert np.all(cold.performance.evaluate("raw_contrast", SEPS) < contrast_floor), (
        "floor must bind"
    )
    np.testing.assert_array_equal(
        np.maximum(seed.performance.evaluate("raw_contrast", SEPS), contrast_floor),
        np.full_like(SEPS, contrast_floor),
    )
    _assert_same_metrics(warm, cold, names=("raw_contrast",))


def test_explicit_diameter_conversion_does_not_change_cache(tmp_path):
    """The record preparation stores contrast on the package's separation grid."""
    path = _yip(tmp_path, "warm")
    from yippy import lod_inscribed_to_circumscribed

    seed = _YIPData(open_yip(path))
    before = seed._perf_filename()
    seed.performance.evaluate(
        "throughput", lod_inscribed_to_circumscribed(5.0, seed.grid.frame, 5.0)
    )
    assert seed._perf_filename() == before
    seed._prepare_record(HalfMaximum(0.3))
    warm = _YIPData(open_yip(path), rule=HalfMaximum(0.3))
    cold = _YIPData(open_yip(_yip(tmp_path, "cold")), rule=HalfMaximum(0.3))
    _assert_same_metrics(warm, cold, names=("raw_contrast",))


# ---------------------------------------------------------------------------
# Aperture parameters
# ---------------------------------------------------------------------------


def test_truncation_cache_distinguishes_leakage_rule(tmp_path):
    """Fails if aperture-mean and azimuthal-ring records alias in the cache."""
    path = _yip(tmp_path, "warm")
    aperture = _YIPData(open_yip(path), rule=HalfMaximum(0.3))
    warm = _YIPData(
        open_yip(path), rule=HalfMaximum(0.3), leakage_rule="azimuthal_ring_mean"
    )
    cold = _YIPData(
        open_yip(_yip(tmp_path, "cold")),
        rule=HalfMaximum(0.3),
        leakage_rule="azimuthal_ring_mean",
    )
    assert aperture._perf_filename() != warm._perf_filename()
    assert warm.performance.leakage_rule == "azimuthal_ring_mean"
    _assert_same_metrics(warm, cold)


def test_fine_radius_difference_does_not_alias(tmp_path):
    """Radii 0.706 and 0.708 share a two-decimal token but not an aperture.

    At ten subpixels per pixel, these radii select different boundary cells
    even though their rounded two-decimal filename tokens would coincide.
    """
    path = _yip(tmp_path, "warm")
    first = _YIPData(open_yip(path), rule=Circular(0.706))
    warm = _YIPData(open_yip(path), rule=Circular(0.708))
    cold = _YIPData(open_yip(_yip(tmp_path, "cold")), rule=Circular(0.708))
    assert not np.allclose(
        first.performance.evaluate("throughput", SEPS),
        cold.performance.evaluate("throughput", SEPS),
        rtol=RTOL,
    )
    assert first._perf_filename() != warm._perf_filename()
    _assert_same_metrics(warm, cold)


def test_fine_threshold_difference_has_distinct_identity(tmp_path):
    """Thresholds 0.301 and 0.304 get distinct performance identities."""
    coro = _YIPData(open_yip(_yip(tmp_path, "yip")))
    coro._prepare_record(HalfMaximum(0.301))
    first = coro._perf_filename()
    coro._prepare_record(HalfMaximum(0.304))
    assert coro._perf_filename() != first


def test_equivalent_scalar_types_share_identity(tmp_path):
    """An integer-valued radius and the equal float resolve to one entry."""
    path = _yip(tmp_path, "yip")
    a = _YIPData(open_yip(path), rule=Circular(1))
    b = _YIPData(open_yip(path), rule=Circular(np.float64(1.0)))
    assert a._perf_filename() == b._perf_filename()


# ---------------------------------------------------------------------------
# Explicit sampling axes
# ---------------------------------------------------------------------------


def test_explicit_separations_do_not_feed_automatic_cache(tmp_path, monkeypatch):
    """Fails if a custom separation grid overwrites the automatic cache identity."""
    path = _yip(tmp_path, "warm")
    obj = _YIPData(open_yip(path))
    filename = obj._perf_filename()
    obj._prepare_record(
        rule=Circular(0.7),
        separations_lod=np.array([0.5, 2.5, 4.5]),
        save_to_fits=True,
        performance_file=filename,
        cache_dir=obj._perf_dir,
    )
    warm = _YIPData(open_yip(path))
    cold = _YIPData(open_yip(_yip(tmp_path, "cold")))
    np.testing.assert_array_equal(
        warm.performance.separation_lod, cold.performance.separation_lod
    )
    _assert_same_metrics(warm, cold)
    _forbid_persisted_computation(monkeypatch)
    _assert_same_metrics(_YIPData(open_yip(path)), cold)


@pytest.mark.parametrize("mode", ["circle", "truncation", "prepared"])
def test_default_construction_measures_the_selected_aperture(tmp_path, mode):
    """Fails if prepared contrast uses an unrelated circular aperture.

    Circular quadrature has a 1e-5 sampling bound from the analytic Gaussian;
    half-maximum area/flux ratios use 1e-6 for independently rounded stored fields.
    """
    package = write_synthetic_yip(tmp_path / "yip")
    rule = HalfMaximum(0.3) if mode == "truncation" else Circular(0.7)
    rules = (rule, HalfMaximum(0.5)) if mode == "prepared" else (rule,)
    coro = open_yip(package.path).prepare(rules=rules)
    record = coro.curves(HalfMaximum(0.5) if mode == "prepared" else rule)
    sep = 3.0
    if mode == "circle":
        rows, cols = np.mgrid[-29:29, -29:29] + 0.5
        weights = reference_circle_weights(rows, cols, 28.0)
        intensity = np.exp(-(rows**2 + cols**2) / (2 * 20**2)) / (2 * np.pi * 2**2)
        planet_flux = (weights * intensity).sum() / 10**2 * package.occulter(sep)
        area_pixels = np.pi * 2.8**2
        assert record.evaluate("throughput", sep) == pytest.approx(
            planet_flux, rel=1e-5
        )
        assert record.evaluate("core_area_lod2", sep) == pytest.approx(
            area_pixels * package.pixscale**2
        )
    else:
        # Uniform stellar light makes the numerator exactly level times area.
        planet_flux = record.evaluate("throughput", sep)
        area_pixels = record.evaluate("core_area_lod2", sep) / package.pixscale**2
        assert not np.isclose(area_pixels, np.pi * 2.8**2, rtol=1e-3)
    assert record.evaluate("raw_contrast", sep) == pytest.approx(
        package.stellar_level * area_pixels / planet_flux, rel=1e-5
    )


# ---------------------------------------------------------------------------
# Hits, legacy entries, and source changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trunc", [None, 0.3])
def test_unchanged_request_is_a_real_cache_hit(tmp_path, monkeypatch, trunc):
    """A repeated request loads the stored table instead of recomputing it."""
    path = _yip(tmp_path, "yip")
    first = _YIPData(
        open_yip(path), rule=Circular(0.7) if trunc is None else HalfMaximum(trunc)
    )
    _forbid_persisted_computation(monkeypatch)
    second = _YIPData(
        open_yip(path), rule=Circular(0.7) if trunc is None else HalfMaximum(trunc)
    )
    _assert_same_metrics(second, first)


def test_prepared_record_repeat_is_a_real_cache_hit(tmp_path, monkeypatch):
    """Switching back to a ratio already computed loads the stored table."""
    coro = _YIPData(open_yip(_yip(tmp_path, "yip")))
    coro._prepare_record(HalfMaximum(0.3))
    expected = _metrics(coro)
    _forbid_persisted_computation(monkeypatch)
    coro._prepare_record(HalfMaximum(0.3))
    for name, values in _metrics(coro).items():
        assert_allclose(values, expected[name], rtol=RTOL, err_msg=name)


def test_entry_without_identity_metadata_is_ignored(tmp_path):
    """A table lacking identity metadata at the automatic path is a miss."""
    path = _yip(tmp_path, "warm")
    probe = _YIPData(open_yip(path))
    poisoned = replace(
        probe.performance,
        throughput=np.full_like(probe.performance.throughput, 0.5),
        raw_contrast=np.full_like(probe.performance.raw_contrast, 1e-3),
    )
    save_coro_performance_to_fits(poisoned, probe._perf_filename(), probe._perf_dir)
    warm = _YIPData(open_yip(path))
    cold = _YIPData(open_yip(_yip(tmp_path, "cold")))
    _assert_same_metrics(warm, cold)


def test_legacy_named_entry_is_ignored(tmp_path):
    """A table under the pre-identity naming scheme is never read."""
    from yippy._precision import dtype_tag
    from yippy._version import __version__

    path = _yip(tmp_path, "warm")
    probe = _YIPData(open_yip(path))
    legacy = (
        f"aper_0.70_v{__version__}_{probe.package.source.digest}_"
        f"{probe.npixels}px_{dtype_tag()}_sampling1.fits"
    )
    save_coro_performance_to_fits(
        replace(
            probe.performance,
            throughput=np.full_like(probe.performance.throughput, 0.5),
        ),
        legacy,
        probe._perf_dir,
    )
    warm = _YIPData(open_yip(path))
    assert warm._perf_filename() != legacy
    _assert_same_metrics(warm, _YIPData(open_yip(_yip(tmp_path, "cold"))))


def test_source_overwrite_invalidates_performance(tmp_path):
    """An in-place package rewrite is recomputed, not served from cache."""
    path = _yip(tmp_path, "warm")
    _YIPData(open_yip(path))
    write_synthetic_yip(path, psf_flux=0.5)
    _bump_mtime(path)
    warm = _YIPData(open_yip(path))
    cold = _YIPData(open_yip(_yip(tmp_path, "cold", psf_flux=0.5)))
    _assert_same_metrics(warm, cold)


def test_source_selection_does_not_alias(tmp_path):
    """Two off-axis files in one directory get separate cache entries."""
    path = _yip(tmp_path, "warm")
    alt_source = _yip(tmp_path, "alt", psf_sigma_lod=0.7)
    (path / "offax_alt.fits").write_bytes((alt_source / "offax_psf.fits").read_bytes())
    _YIPData(open_yip(path))
    warm = _YIPData(open_yip(path, filenames={"offaxis_psfs": "offax_alt.fits"}))
    cold = _YIPData(open_yip(alt_source))
    _assert_same_metrics(warm, cold)


def test_downsampled_resolution_does_not_alias(tmp_path):
    """A downsampled construction does not reuse the native-resolution table."""
    path = _yip(tmp_path, "warm")
    native = _YIPData(open_yip(path))
    warm = _YIPData(open_yip(path), downsample_shape=(32, 32))
    cold = _YIPData(open_yip(_yip(tmp_path, "cold")), downsample_shape=(32, 32))
    assert native._perf_filename() != warm._perf_filename()
    _assert_same_metrics(warm, cold)


def test_float64_process_uses_separate_entries(tmp_path):
    """A float64 process neither reads nor overwrites the float32 entry."""
    path = _yip(tmp_path, "yip")
    f32 = _YIPData(open_yip(path))
    f32_file = f32._perf_dir / f32._perf_filename()
    f32_bytes = f32_file.read_bytes()
    script = (
        "import json, logging, sys\n"
        "logging.disable(logging.CRITICAL)\n"
        "import jax\n"
        "assert jax.config.jax_enable_x64\n"
        "import numpy as np\n"
        "from yippy._yip import _YIPData\n"
        "from yippy import open_yip\n"
        "c = _YIPData(open_yip(sys.argv[1]))\n"
        "seps = np.array([float(a) for a in sys.argv[2:]])\n"
        "print(json.dumps({'name': c._perf_filename(),"
        " 'thr': [float(v) for v in c.performance.evaluate('throughput', seps)]}))\n"
    )
    env = dict(os.environ, JAX_ENABLE_X64="1")
    out = subprocess.run(
        [sys.executable, "-c", script, str(path), *map(str, SEPS)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["name"] != f32_file.name
    assert (f32._perf_dir / result["name"]).exists()
    assert f32_file.read_bytes() == f32_bytes
    assert_allclose(
        result["thr"], f32.performance.evaluate("throughput", SEPS), rtol=1e-5
    )


# ---------------------------------------------------------------------------
# Scene operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("kind", "warned"), [("1d", True), ("2dq", False)])
def test_scene_operator_warning_names_1d_sampling_only(tmp_path, caplog, kind, warned):
    """Fails if a 1D scene operator stops reporting that its PSFs are not rotated."""
    path = _yip(tmp_path, f"warn_{kind}", kind=kind, npix=16, max_offset_lod=2.0)
    with caplog.at_level("WARNING", logger="yippy"):
        open_yip(path).prepare().with_scene_operator()
    assert any("not rotated" in r.getMessage() for r in caplog.records) is warned


def test_synthesized_core_cache_does_not_feed_default_curves(tmp_path):
    """Opt-in synthesized core tables cannot masquerade as native-offset tables."""
    path = _yip(tmp_path, "warm", offset_step_lod=2.0)
    obj = _YIPData(open_yip(path), rule=HalfMaximum(0.5))
    expected = _metrics(obj)
    obj._prepare_record(
        rule=HalfMaximum(0.5),
        separations_lod=np.arange(0.5, 6.5, 0.5),
        performance_file=obj._perf_filename(),
        cache_dir=obj._perf_dir,
    )
    warm = _YIPData(open_yip(path), rule=HalfMaximum(0.5))
    for name, value in _metrics(warm).items():
        assert_allclose(value, expected[name], rtol=RTOL, err_msg=name)


@pytest.mark.parametrize("trunc", [None, 0.3])
def test_legacy_circle_method_is_not_reused(tmp_path, trunc):
    """A table labeled with the previous cubic aperture cannot supply new curves."""
    path = _yip(tmp_path, "warm")
    obj = _YIPData(
        open_yip(path), rule=Circular(0.7) if trunc is None else HalfMaximum(trunc)
    )
    expected = _metrics(obj)
    identity = obj._auto_perf_identity()
    identity.pop("circle_method", None)
    identity.pop("circle_oversample", None)
    identity["oversample"] = 2
    if trunc is not None:
        identity["core_step_lod"] = 0.5
    save_coro_performance_to_fits(
        replace(
            obj.performance,
            throughput=np.full_like(obj.performance.throughput, 0.123),
            raw_contrast=np.full_like(obj.performance.raw_contrast, 0.456),
        ),
        obj._perf_filename(),
        obj._perf_dir,
        identity=identity,
    )
    warm = _YIPData(
        open_yip(path), rule=Circular(0.7) if trunc is None else HalfMaximum(trunc)
    )
    for name, value in _metrics(warm).items():
        assert_allclose(value, expected[name], rtol=RTOL, err_msg=name)


@pytest.mark.parametrize("trunc", [None, 0.3])
def test_center_counting_method_is_not_reused(tmp_path, trunc):
    """A center-counting table cannot supply fractional, source-local curves."""
    path = _yip(tmp_path, "warm")
    obj = _YIPData(
        open_yip(path), rule=Circular(0.7) if trunc is None else HalfMaximum(trunc)
    )
    expected = _metrics(obj)
    identity = obj._auto_perf_identity()
    identity["circle_method"] = "band_limited_float64_subpixel_centers"
    save_coro_performance_to_fits(
        replace(
            obj.performance,
            throughput=np.full_like(obj.performance.throughput, 0.123),
            raw_contrast=np.full_like(obj.performance.raw_contrast, 0.456),
        ),
        obj._perf_filename(),
        obj._perf_dir,
        identity=identity,
    )
    warm = _YIPData(
        open_yip(path), rule=Circular(0.7) if trunc is None else HalfMaximum(trunc)
    )
    for name, value in _metrics(warm).items():
        assert_allclose(value, expected[name], rtol=RTOL, err_msg=name)


def test_performance_fits_preserves_every_float64_bit(tmp_path):
    """Fails if FITS rounds any performance array or loses aperture metadata."""
    separation = np.array([5.123456789012345, 10.123456789012345])
    throughput = np.array([0.17721052431234567, 0.28832163542345678])
    contrast = np.array([1.234567890123456e-11, 2.345678901234567e-10])
    record = perf.PerformanceCurves(
        separation_lod=separation,
        frame=Frame(
            reference_wavelength_nm=500.0,
            diameter_m=6.0,
            diameter_convention="circumscribed",
        ),
        rule=HalfMaximum(0.304),
        throughput=throughput,
        throughput_variance=0.5 * throughput,
        core_area_lod2=np.array([1.123456789012345, 2.123456789012345]),
        core_area_variance_lod2=0.5 * np.array([1.123456789012345, 2.123456789012345]),
        leakage_fraction=contrast * throughput,
        leakage_fraction_variance=0.5 * contrast * throughput,
        leakage_density_lod2=contrast,
        leakage_density_variance_lod2=0.5 * contrast,
        raw_contrast=contrast,
        iwa_lod=1.23456789012345,
        data_extent_lod=16.0,
        leakage_rule="azimuthal_ring_mean",
        stellar_diameter_lod=0.123456789012345,
    )
    save_coro_performance_to_fits(record, "performance.fits", tmp_path)
    loaded = load_coro_performance_from_fits("performance.fits", tmp_path)
    for name in (
        "separation_lod",
        "throughput",
        "throughput_variance",
        "core_area_lod2",
        "core_area_variance_lod2",
        "leakage_fraction",
        "leakage_fraction_variance",
        "leakage_density_lod2",
        "leakage_density_variance_lod2",
        "raw_contrast",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(record, name))
    for name in (
        "rule",
        "frame",
        "leakage_rule",
        "stellar_diameter_lod",
        "iwa_lod",
        "data_extent_lod",
    ):
        assert getattr(loaded, name) == getattr(record, name)
