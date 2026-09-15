"""Cached results must not depend on the history of earlier calculations.

Each test compares an object built after some earlier call sequence (warm)
against an object built from an identical package written to a separate
directory, which cannot share cache entries with it (cold). The two must agree
to the precision of the stored table, and a cache entry written under different
settings must never be reused.

Tolerance basis: the performance table is stored as single precision FITS
columns (format ``E``), so a value read back differs from the value computed in
the same process by up to one float32 unit roundoff (about 6e-8 relative). The
spline evaluation adds roundoff of the same order. ``RTOL = 1e-6`` leaves an
order of magnitude of margin while staying far below any physical difference
(the defects these tests guard against differ at the 1e-3 to 1e-1 level).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from synthetic_yip import write_synthetic_yip

import yippy.coronagraph as coronagraph_module
import yippy.performance as perf
from yippy import Coronagraph
from yippy.util import save_coro_performance_to_fits

RTOL = 1e-6
SEPS = np.array([1.5, 3.0, 4.25])


def _yip(tmp_path, name, **kwargs):
    return write_synthetic_yip(tmp_path / name, **kwargs).path


def _metrics(coro):
    return {
        "throughput": np.asarray(coro.throughput(SEPS)),
        "core_area": np.asarray(coro.core_area(SEPS)),
        "raw_contrast": np.asarray(coro.raw_contrast(SEPS)),
    }


def _assert_same_metrics(warm, cold, names=("throughput", "core_area", "raw_contrast")):
    got, want = _metrics(warm), _metrics(cold)
    for name in names:
        assert_allclose(got[name], want[name], rtol=RTOL, err_msg=name)


def _patch_everywhere(monkeypatch, name, replacement):
    """Replace a performance function in every module that imported it."""
    for module in (perf, coronagraph_module):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, replacement)


def _forbid_persisted_computation(monkeypatch):
    """Make any recomputation of a persisted quantity raise."""

    def fail(*args, **kwargs):
        raise AssertionError("persisted quantity was recomputed")

    for name in (
        "compute_throughput_curve",
        "compute_truncation_throughput_curve",
        "compute_raw_contrast_curve",
    ):
        _patch_everywhere(monkeypatch, name, fail)


def _bump_mtime(directory):
    """Move every source file's mtime forward so the change is unambiguous."""
    for path in Path(directory).glob("*.fits"):
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


# ---------------------------------------------------------------------------
# Truncation setter and constructor
# ---------------------------------------------------------------------------


def test_setter_then_constructor_matches_cold(tmp_path):
    """A cache written by the setter reproduces a cold truncation construction."""
    path = _yip(tmp_path, "warm")
    seed = Coronagraph(path)
    seed.set_psf_trunc_ratio(0.3)
    warm = Coronagraph(path, psf_trunc_ratio=0.3)
    cold = Coronagraph(_yip(tmp_path, "cold"), psf_trunc_ratio=0.3)
    _assert_same_metrics(warm, cold)
    # The setter object itself must also match
    _assert_same_metrics(seed, cold)


def test_constructor_then_setter_matches_cold(tmp_path):
    """Reaching truncation mode through the setter after a cache hit matches cold."""
    path = _yip(tmp_path, "warm")
    Coronagraph(path, psf_trunc_ratio=0.3)
    warm = Coronagraph(path)
    warm.set_psf_trunc_ratio(0.3)
    cold = Coronagraph(_yip(tmp_path, "cold"), psf_trunc_ratio=0.3)
    _assert_same_metrics(warm, cold)


def test_setter_does_not_persist_contrast_floor(tmp_path):
    """The setter stores unfloored contrast, as the constructor does."""
    path = _yip(tmp_path, "warm")
    seed = Coronagraph(path, contrast_floor=1e-8)
    seed.set_psf_trunc_ratio(0.3)
    warm = Coronagraph(path, psf_trunc_ratio=0.3)
    cold = Coronagraph(_yip(tmp_path, "cold"), psf_trunc_ratio=0.3)
    assert np.all(cold.raw_contrast(SEPS) < 1e-8), "floor must bind for this test"
    _assert_same_metrics(warm, cold, names=("raw_contrast",))


def test_setter_does_not_persist_inscribed_diameter_scaling(tmp_path):
    """The setter stores contrast on the package's separation grid."""
    path = _yip(tmp_path, "warm")
    seed = Coronagraph(path, use_inscribed_diameter=True)
    seed.set_psf_trunc_ratio(0.3)
    warm = Coronagraph(path, psf_trunc_ratio=0.3)
    cold = Coronagraph(_yip(tmp_path, "cold"), psf_trunc_ratio=0.3)
    _assert_same_metrics(warm, cold, names=("raw_contrast",))


# ---------------------------------------------------------------------------
# Aperture parameters
# ---------------------------------------------------------------------------


def test_truncation_cache_distinguishes_contrast_aperture(tmp_path):
    """Truncation-mode entries at radii 0.7 and 0.85 do not alias."""
    path = _yip(tmp_path, "warm")
    Coronagraph(path, psf_trunc_ratio=0.3, aperture_radius_lod=0.7)
    warm = Coronagraph(path, psf_trunc_ratio=0.3, aperture_radius_lod=0.85)
    cold = Coronagraph(
        _yip(tmp_path, "cold"), psf_trunc_ratio=0.3, aperture_radius_lod=0.85
    )
    _assert_same_metrics(warm, cold)


def test_fine_radius_difference_does_not_alias(tmp_path):
    """Radii 0.706 and 0.708 share a two-decimal token but not an aperture.

    At the default oversampling the aperture radius is ``8 r`` oversampled
    pixels and a pixel lies at distance ``sqrt(32) = 5.657`` from the peak, so
    that pixel is outside the 0.706 aperture (5.648) and inside 0.708 (5.664).
    """
    path = _yip(tmp_path, "warm")
    first = Coronagraph(path, aperture_radius_lod=0.706)
    warm = Coronagraph(path, aperture_radius_lod=0.708)
    cold = Coronagraph(_yip(tmp_path, "cold"), aperture_radius_lod=0.708)
    assert not np.allclose(first.throughput(SEPS), cold.throughput(SEPS), rtol=RTOL)
    assert first._perf_filename() != warm._perf_filename()
    _assert_same_metrics(warm, cold)


def test_fine_threshold_difference_has_distinct_identity(tmp_path):
    """Thresholds 0.301 and 0.304 get distinct performance identities."""
    coro = Coronagraph(_yip(tmp_path, "yip"))
    coro.psf_trunc_ratio = 0.301
    first = coro._perf_filename()
    coro.psf_trunc_ratio = 0.304
    assert coro._perf_filename() != first


def test_equivalent_scalar_types_share_identity(tmp_path):
    """An integer-valued radius and the equal float resolve to one entry."""
    path = _yip(tmp_path, "yip")
    a = Coronagraph(path, aperture_radius_lod=1)
    b = Coronagraph(path, aperture_radius_lod=np.float64(1.0))
    assert a._perf_filename() == b._perf_filename()


# ---------------------------------------------------------------------------
# Oversampling
# ---------------------------------------------------------------------------


def test_explicit_oversampling_does_not_feed_automatic_cache(tmp_path, monkeypatch):
    """A table saved at oversample 5 under the automatic name is not reused.

    A table saved at the automatic settings is.
    """
    path = _yip(tmp_path, "warm")
    obj = Coronagraph(path)
    obj.compute_all_performance_curves(
        aperture_radius_lod=0.7,
        oversample=5,
        save_to_fits=True,
        performance_file=obj._perf_filename(),
        cache_dir=obj._perf_dir,
    )
    warm = Coronagraph(path)
    cold = Coronagraph(_yip(tmp_path, "cold"))
    _assert_same_metrics(warm, cold)

    obj.compute_all_performance_curves(
        aperture_radius_lod=0.7,
        oversample=2,
        save_to_fits=True,
        performance_file=obj._perf_filename(),
        cache_dir=obj._perf_dir,
    )
    _forbid_persisted_computation(monkeypatch)
    _assert_same_metrics(Coronagraph(path), cold)


def test_default_construction_uses_oversample_two(tmp_path, monkeypatch):
    """The cold constructor keeps its oversampling factor of 2 in both modes."""
    seen = []
    for name in (
        "compute_throughput_curve",
        "compute_truncation_throughput_curve",
        "compute_truncation_core_area_curve",
        "compute_raw_contrast_curve",
    ):
        original = getattr(perf, name)

        def spy(*args, _original=original, _name=name, **kwargs):
            seen.append((_name, kwargs.get("oversample")))
            return _original(*args, **kwargs)

        _patch_everywhere(monkeypatch, name, spy)
    path = _yip(tmp_path, "yip")
    Coronagraph(path)
    Coronagraph(path, psf_trunc_ratio=0.3)
    coro = Coronagraph(_yip(tmp_path, "setter"))
    coro.set_psf_trunc_ratio(0.5)
    assert seen, "no calculation observed"
    assert all(factor == 2 for _, factor in seen), seen


# ---------------------------------------------------------------------------
# Hits, legacy entries, and source changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trunc", [None, 0.3])
def test_unchanged_request_is_a_real_cache_hit(tmp_path, monkeypatch, trunc):
    """A repeated request loads the stored table instead of recomputing it."""
    path = _yip(tmp_path, "yip")
    first = Coronagraph(path, psf_trunc_ratio=trunc)
    _forbid_persisted_computation(monkeypatch)
    second = Coronagraph(path, psf_trunc_ratio=trunc)
    _assert_same_metrics(second, first)


def test_setter_repeat_is_a_real_cache_hit(tmp_path, monkeypatch):
    """Switching back to a ratio already computed loads the stored table."""
    coro = Coronagraph(_yip(tmp_path, "yip"))
    coro.set_psf_trunc_ratio(0.3)
    expected = _metrics(coro)
    _forbid_persisted_computation(monkeypatch)
    coro.set_psf_trunc_ratio(0.3)
    for name, values in _metrics(coro).items():
        assert_allclose(values, expected[name], rtol=RTOL, err_msg=name)


def test_entry_without_identity_metadata_is_ignored(tmp_path):
    """A table lacking identity metadata at the automatic path is a miss."""
    path = _yip(tmp_path, "warm")
    probe = Coronagraph(path)
    sep = np.arange(0.0, 6.5, 0.5)
    save_coro_performance_to_fits(
        sep,
        np.full_like(sep, 0.5),
        np.full_like(sep, 1e-3),
        probe._perf_filename(),
        probe._perf_dir,
    )
    warm = Coronagraph(path)
    cold = Coronagraph(_yip(tmp_path, "cold"))
    _assert_same_metrics(warm, cold)


def test_legacy_named_entry_is_ignored(tmp_path):
    """A table under the pre-identity naming scheme is never read."""
    from yippy._precision import dtype_tag
    from yippy._version import __version__

    path = _yip(tmp_path, "warm")
    probe = Coronagraph(path)
    sep = np.arange(0.0, 6.5, 0.5)
    legacy = (
        f"aper_0.70_v{__version__}_{probe._source_signature()}_"
        f"{probe.npixels}px_{dtype_tag()}_sampling1.fits"
    )
    save_coro_performance_to_fits(
        sep, np.full_like(sep, 0.5), np.full_like(sep, 1e-3), legacy, probe._perf_dir
    )
    warm = Coronagraph(path)
    assert warm._perf_filename() != legacy
    _assert_same_metrics(warm, Coronagraph(_yip(tmp_path, "cold")))


def test_source_overwrite_invalidates_performance(tmp_path):
    """An in-place package rewrite is recomputed, not served from cache."""
    path = _yip(tmp_path, "warm")
    Coronagraph(path)
    write_synthetic_yip(path, psf_flux=0.5)
    _bump_mtime(path)
    warm = Coronagraph(path)
    cold = Coronagraph(_yip(tmp_path, "cold", psf_flux=0.5))
    _assert_same_metrics(warm, cold)


def test_source_selection_does_not_alias(tmp_path):
    """Two off-axis files in one directory get separate cache entries."""
    path = _yip(tmp_path, "warm")
    alt_source = _yip(tmp_path, "alt", psf_sigma_lod=0.7)
    (path / "offax_alt.fits").write_bytes((alt_source / "offax_psf.fits").read_bytes())
    Coronagraph(path)
    warm = Coronagraph(path, offax_data_file="offax_alt.fits")
    cold = Coronagraph(alt_source)
    _assert_same_metrics(warm, cold)


def test_downsampled_resolution_does_not_alias(tmp_path):
    """A downsampled construction does not reuse the native-resolution table."""
    path = _yip(tmp_path, "warm")
    native = Coronagraph(path)
    warm = Coronagraph(path, downsample_shape=(32, 32))
    cold = Coronagraph(_yip(tmp_path, "cold"), downsample_shape=(32, 32))
    assert native._perf_filename() != warm._perf_filename()
    _assert_same_metrics(warm, cold)


def test_float64_process_uses_separate_entries(tmp_path):
    """A float64 process neither reads nor overwrites the float32 entry."""
    path = _yip(tmp_path, "yip")
    f32 = Coronagraph(path)
    f32_file = f32._perf_dir / f32._perf_filename()
    f32_bytes = f32_file.read_bytes()
    script = (
        "import json, logging, sys\n"
        "logging.disable(logging.CRITICAL)\n"
        "import jax\n"
        "assert jax.config.jax_enable_x64\n"
        "import numpy as np\n"
        "from yippy import Coronagraph\n"
        "c = Coronagraph(sys.argv[1])\n"
        "seps = np.array([float(a) for a in sys.argv[2:]])\n"
        "print(json.dumps({'name': c._perf_filename(),"
        " 'thr': [float(v) for v in c.throughput(seps)]}))\n"
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
    assert_allclose(result["thr"], f32.throughput(SEPS), rtol=1e-5)


# ---------------------------------------------------------------------------
# PSF datacubes
# ---------------------------------------------------------------------------


def _small_2dq(tmp_path, name, **kwargs):
    return _yip(tmp_path, name, kind="2dq", npix=16, max_offset_lod=2.0, **kwargs)


def _cube(coro):
    coro.create_psf_datacube()
    return np.asarray(coro.psf_datacube)


def _assert_same_cube(warm, cold):
    """Two independently synthesized cubes agree to synthesis roundoff.

    Tolerance basis: XLA's multithreaded CPU kernels can order the FFT and
    neighbor-sum additions differently between runs, moving pixels by up to
    one unit roundoff of the peak (about 6e-8 of it in float32). ``1e-6`` of
    the peak leaves margin for that; the aliasing defects guarded here change
    the cube by tens of percent of the peak.
    """
    got, want = _cube(warm), _cube(cold)
    assert_allclose(got, want, rtol=0, atol=1e-6 * want.max())


def test_datacube_identity_includes_physical_symmetry(tmp_path):
    """Full cubes built with different x symmetry do not alias."""
    path = _small_2dq(tmp_path, "warm")
    _cube(Coronagraph(path, use_quarter_psf_datacube=False))
    warm = Coronagraph(path, use_quarter_psf_datacube=False, x_symmetric=False)
    cold = Coronagraph(
        _small_2dq(tmp_path, "cold"), use_quarter_psf_datacube=False, x_symmetric=False
    )
    _assert_same_cube(warm, cold)


def test_quarter_and_full_cubes_are_distinct(tmp_path):
    """Quarter and full storage use separate entries."""
    path = _small_2dq(tmp_path, "yip")
    quarter = Coronagraph(path)
    full = Coronagraph(path, use_quarter_psf_datacube=False)
    assert quarter._datacube_cache_path != full._datacube_cache_path
    assert _cube(quarter).shape != _cube(full).shape


def test_gpu_path_loads_the_cpu_entry(tmp_path, monkeypatch):
    """The GPU loader reads the same identity the CPU writer produced."""
    path = _small_2dq(tmp_path, "yip")
    expected = _cube(Coronagraph(path))
    other = Coronagraph(path)

    def fail(*args, **kwargs):
        raise AssertionError("cube was regenerated")

    monkeypatch.setattr(other.offax, "create_psfs", fail)
    other._create_psf_datacube_gpu()
    assert_allclose(np.asarray(other.psf_datacube), expected, rtol=0, atol=0)


def test_source_overwrite_invalidates_datacube(tmp_path):
    """A rewritten package gets a fresh cube."""
    path = _small_2dq(tmp_path, "warm")
    _cube(Coronagraph(path))
    write_synthetic_yip(path, kind="2dq", npix=16, max_offset_lod=2.0, psf_flux=0.5)
    _bump_mtime(path)
    warm = Coronagraph(path)
    cold = Coronagraph(_small_2dq(tmp_path, "cold", psf_flux=0.5))
    _assert_same_cube(warm, cold)


def test_stale_object_does_not_label_cube_as_new_source(tmp_path):
    """An object loaded before a source edit writes under its original identity."""
    path = _small_2dq(tmp_path, "warm")
    stale = Coronagraph(path)
    write_synthetic_yip(path, kind="2dq", npix=16, max_offset_lod=2.0, psf_flux=0.5)
    _bump_mtime(path)
    _cube(stale)
    warm = Coronagraph(path)
    cold = Coronagraph(_small_2dq(tmp_path, "cold", psf_flux=0.5))
    _assert_same_cube(warm, cold)
