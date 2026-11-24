"""Optional off-axis routing, geometry and refusal regression tests."""

import copy
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from footprint_truth import assert_stellar_read
from hwoutils.memory import MemoryLimit
from lod_unit import lod
from optixstuff.grids import Image, Measure, PixelGrid, resample
from sinc_fixture import save_manifest, write_delivery
from synthetic_yip import write_synthetic_yip

import yippy.sinc as sc
from yippy import SincOptions, open_yip
from yippy._util import cache_identity_digest
from yippy._yip import _YIPData
from yippy.delivery.format import detect_delivery, file_sha256, load_delivery
from yippy.delivery.routing import route_interpolation
from yippy.export import export_exosims
from yippy.performance import stellar_intensity_curves
from yippy.scene import AbstractSceneOperator
from yippy.sinc.evidence import Reason, Status


@pytest.fixture(scope="module")
def admitted(tmp_path_factory):
    """One tiny admitted quadrant and real performance cache for all reads."""
    with jax.enable_x64():
        root = tmp_path_factory.mktemp("sinc")
        metadata = write_delivery(root)
        # Independent-check intake has its own test. Keep the shared reader's
        # cold setup small by admitting only the mandatory mirror evidence.
        metadata["checks"] = []
        metadata["served"]["bounds_lod"] = [[-0.2, 0.2], [-0.2, 0.5]]
        save_manifest(root, metadata)
        with fits.open(root / "legacy" / "stellar_intens.fits", mode="update") as hdus:
            columns = np.arange(4) - 1.5
            hdus[0].data *= 1 + 0.2 * np.cos(2 * np.pi * columns / 4)

        def reject_legacy(*args, **kwargs):
            raise AssertionError("versioned delivery entered legacy OffJAX intake")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("yippy._yip.OffJAX", reject_legacy)
            coro = open_yip(root).prepare(interpolation="sinc")
        yield coro
    jax.clear_caches()


def test_unmarked_blend_and_auto(tmp_path):
    """Legacy auto uses the blend images and flags unsupported layout."""
    fixture = write_synthetic_yip(
        tmp_path / "legacy", npix=4, max_offset_lod=1.0, offset_step_lod=0.5
    )
    package = open_yip(fixture.path)
    with pytest.raises(ValueError, match="product_id"):
        open_yip(fixture.path, product_id="mono")
    blend = package.prepare()
    auto = package.prepare(interpolation="auto")
    assert auto.interpolation_info.reasons == ("LAYOUT_UNSUPPORTED",)
    np.testing.assert_array_equal(
        blend.source_psfs(None, 0.3, 0.2).image.data,
        auto.source_psfs(None, 0.3, 0.2).image.data,
    )
    assert not auto.source_psfs(None, np.nan, 0.0).valid[0]
    with pytest.raises(ValueError, match="versioned"):
        package.prepare(interpolation="sinc")
    with pytest.raises(ValueError, match="Bounded policies require sinc"):
        package.prepare(
            interpolation="blend",
            sinc_options=SincOptions(policy=SimpleNamespace(mode="bounded")),
        )


@pytest.mark.parametrize("defect", ["schema", "hash", "header", "duplicate", "hole"])
def test_delivery_corruption(tmp_path, defect):
    """Fails if auto hides malformed declarations or corrupt measured data."""
    m = write_delivery(tmp_path)
    if defect == "schema":
        m["schema"] = "sinc-delivery-99"
    elif defect == "hash":
        m["planes"][0]["psf_sha256"] = "0" * 64
    elif defect == "header":
        with fits.open(tmp_path / "nodes.fits", mode="update") as hdus:
            hdus[0].header["PIXSCALE"] = 0.2
        m["planes"][0]["psf_sha256"] = file_sha256(tmp_path / "nodes.fits")
    else:
        path = tmp_path / "training.csv"
        rows = path.read_text().splitlines()
        row = rows[-1].split(",")
        if defect == "duplicate":
            row[1:3] = ["0", "0"]
        else:
            row[1] = "18"
        rows[-1] = ",".join(row)
        path.write_text("\n".join(rows) + "\n")
    save_manifest(tmp_path, m)
    with pytest.raises(ValueError):
        route_interpolation(tmp_path, "auto", None, None)


def test_full_blend_photon_conversion(tmp_path):
    """Measured mean densities receive the photon and pixel-area factors once."""
    metadata = write_delivery(tmp_path, full=True)
    metadata["normalization"]["kind"] = "unocculted_peak"
    metadata["planes"][0]["photon_conversion"] = 2.0
    metadata["sample_kind"] = "mean"
    save_manifest(tmp_path, metadata)
    delivery, blend, _ = route_interpolation(tmp_path, "blend", None, None)
    assert delivery.measure == Measure.integral()
    expected = fits.getdata(tmp_path / "nodes.fits")[16 * 33 + 16] * (2.0 * 0.25**2)
    np.testing.assert_allclose(blend.create_psf(0.0, 0.0).data, expected, rtol=2e-7)


def test_product_selection(tmp_path):
    """Fails if a multi-plane manifest silently selects by list position."""
    m = write_delivery(tmp_path)
    m["diameter_convention"] = "inscribed"
    m["planes"].append(dict(m["planes"][0], id="other"))
    m["spectral_sampling"]["node_count"] = 2
    save_manifest(tmp_path, m)
    assert detect_delivery(tmp_path)["layout"] == "quadrant_rectangle"
    with pytest.raises(ValueError, match="product_id"):
        load_delivery(tmp_path, m, SincOptions())
    # A selected plane must not open another product's files.
    m["planes"][0]["psf_path"] = "nodes.fits"
    with fits.open(tmp_path / "nodes.fits", mode="update") as hdus:
        hdus[0].header["LAMBDA"] = (0.5, "um")
        hdus[0].header["D"] = (600.0, "cm")
        hdus[0].header["PIXSCALE"] = (m["pixel_scale_arcsec_xy"][0], "arcsec/pixel")
    m["planes"][0]["psf_sha256"] = file_sha256(tmp_path / "nodes.fits")
    m["planes"][1]["psf_sha256"] = file_sha256(tmp_path / "nodes.fits")
    save_manifest(tmp_path, m)
    package = open_yip(tmp_path, product_id="other")
    from yippy import write_yip

    copied = open_yip(write_yip(tmp_path / "copied", package), product_id="other")
    assert copied.source.digest == package.source.digest
    assert copied.fits_header("offaxis_psfs") == package.fits_header("offaxis_psfs")
    chosen = load_delivery(
        copied.source.path,
        detect_delivery(copied.source.path),
        SincOptions(product_id="other"),
        package=copied,
    )
    assert package.header.offaxis_psfs.pixel_grid() == chosen.grid
    assert chosen.grid.frame.diameter_convention == "inscribed"
    np.testing.assert_array_equal(package.offaxis_psfs / 0.25**2, chosen.nodes)
    np.testing.assert_array_equal(package.offsets_xy_lod, chosen.offsets)
    assert chosen.header.pixscale.value == 0.25
    np.testing.assert_allclose(chosen.header.lambda0.to_value("nm"), 500.0, atol=1e-12)
    again = load_delivery(
        copied.source.path,
        copied._manifest,
        SincOptions(product_id="other"),
        package=copied,
    )
    np.testing.assert_array_equal(chosen.nodes, again.nodes)
    np.testing.assert_array_equal(copied.offaxis_psfs / 0.25**2, again.nodes)
    assert chosen.plane["id"] == "other"
    assert chosen.nodes.dtype == np.float32
    bad_check = dict(m["checks"][0], image_index=-1)
    with pytest.raises(ValueError, match="image_index"):
        chosen.read_images([bad_check])


def test_quadrant_native_intake(admitted):
    """Fails if manifest intake applies legacy center, mirrors or float casting."""
    grid = admitted._offaxis_backend.reader.grid
    assert admitted.bandpass is None
    assert grid.products.measure == Measure.density("grid")
    assert grid.intensity.dtype == jnp.float32
    assert grid.intensity.shape == (1, 18, 18, 4, 4)
    np.testing.assert_array_equal(
        (grid.products.pixels.center_x, grid.products.pixels.center_y), [1.5, 1.5]
    )
    assert (
        admitted._offaxis_backend.get_psf_by_offset_idx(0, 0).reconstruction
        == "band_limited"
    )


def test_reads_units_refusals(admitted):
    """Fails on integral/mean conversion, clipped sinc output or NaN acceptance."""
    reader = admitted._offaxis_backend.reader
    direct = reader.read(jnp.array([0.07, -0.04]), jnp.int32(0))
    actual = jax.tree.map(
        lambda value: value[0],
        admitted._offaxis_backend.create_psfs_with_info([0.07], [-0.04]),
    )
    np.testing.assert_allclose(actual.intensity, direct.intensity * 0.25**2, atol=1e-18)
    assert actual.selected_backend == 1
    assert actual.ledger.target_id.endswith(
        ":" + admitted.interpolation_info.product_id
    )
    assert actual.ledger.target_id.startswith("incident_photon_per_pixel:")
    # Replacing leaves here tests the conversion operator only. Public reuse
    # requires fresh admission and evidence for changed samples.
    signed_backend = eqx.tree_at(
        lambda b: b.reader.grid.intensity,
        admitted._offaxis_backend,
        -admitted._offaxis_backend.reader.grid.intensity,
    )
    np.testing.assert_allclose(
        signed_backend.create_psf(0.07, -0.04).data, -direct.intensity, atol=1e-18
    )
    refused = jax.tree.map(
        lambda value: value[0],
        admitted._offaxis_backend.create_psfs_with_info([np.nan], [0.0]),
    )
    assert not refused.accepted
    assert int(refused.reason_bits) & int(Reason.NONFINITE_QUERY)
    assert not admitted.source_psfs(None, np.nan, 0.0).valid[0]
    # A refused source is flagged and its image is exactly zero, not NaN.
    assert (
        eqx.filter_jit(admitted.source_psfs)(None, np.nan, 0.0).image.data == 0.0
    ).all()
    batch = admitted.source_psfs(
        None, jnp.array([0.07, np.nan]), jnp.array([-0.04, 0.0])
    )
    np.testing.assert_array_equal(batch.valid, [True, False])
    assert admitted.source_psfs(None, jnp.empty(0), jnp.empty(0)).image.data.shape == (
        0,
        4,
        4,
    )
    assert (
        admitted.source_psfs(None, 0.07, -0.04).image.reconstruction == "band_limited"
    )
    assert (
        admitted.source_psfs(
            None, jnp.array([0.07]), jnp.array([-0.04])
        ).image.reconstruction
        == "band_limited"
    )


def test_stellar_frame_and_eqx(admitted, tmp_path, monkeypatch):
    """Fails if stellar pitch is overwritten or reader tables hide in closures.

    The Fourier oracle below checks pitch and frame handling, so the stellar map is
    resampled band-limited here; the shipped default is linear (sampling.py).
    The angular sky-response read is the reference for the reframed profile,
    with float64 interpolation roundoff. This catches using the native radii.
    """
    import yippy._stellar_intens as stellar_intens
    import yippy.coronagraph as coronagraph

    monkeypatch.setattr(coronagraph, "STELLAR_RECONSTRUCTION", "band_limited")
    monkeypatch.setattr(stellar_intens, "STELLAR_RECONSTRUCTION", "band_limited")
    view = admitted
    host = _YIPData(admitted.package, interpolation="sinc")
    assert np.ptp(view.stellar_map(None, 0.0).data) > 0
    # Independent Fourier series integrated over target pixel boxes.
    # The 1e-14 bound allows float64 DFT and exponential roundoff.
    # Zero extension doubles each axis (period 8): the table is centered in a
    # zero field, since nothing is reflected past its edge.
    stellar = view.stellar_map(None, 0.0)
    data = np.asarray(stellar.integrated().data)
    even = np.zeros((8, 8))
    even[2:6, 2:6] = data
    spectrum = np.fft.fft2(even)
    frequencies = np.fft.fftfreq(8)
    for pitch in (0.3, 0.4):
        ratio = pitch / 0.4
        positions = (np.arange(4) - 1.5) * ratio + 3.5
        phase = 2 * np.pi * positions[:, None] * frequencies
        basis = np.exp(1j * phase)
        basis[:, 4] = np.cos(phase[:, 4])
        basis *= np.sinc(frequencies * ratio)
        expected = (basis @ spectrum @ basis.T).real / 8**2 * ratio**2
        sampled = view.stellar_map(
            500.0,
            0.0,
            grid=PixelGrid.regular((4, 4), pitch, "lod", frame=view.stellar_grid.frame),
        )
        assert sampled.reconstruction == "band_limited"
        np.testing.assert_allclose(sampled.data, expected, rtol=1e-14, atol=0)
        wrong = expected * (0.4 / 0.25) ** 2
        assert not np.allclose(expected, wrong, atol=0.0, rtol=1e-3)
    assert view.stellar_grid.step == 0.4
    assert view.stellar_grid.frame.reference_wavelength_nm == 400.0
    assert view.stellar_grid.frame.diameter_m == 6.0
    assert view.grid.frame.reference_wavelength_nm == pytest.approx(500.0)
    assert view.grid.step == 0.25
    assert view.header.pixscale.value == 0.4
    separations = np.array([0.2, 0.3])
    exported = export_exosims(
        view, "arcsec", separations_lod=separations, output_dir=tmp_path
    )
    assert exported["starlightSuppressionSystems"][0]["lam"] == 500.0
    sky = fits.getdata(tmp_path / "occ_trans.fits")
    np.testing.assert_allclose(
        sky[:, 0], separations * view.grid.frame.lod_to_arcsec(500), rtol=1e-14
    )
    np.testing.assert_allclose(
        sky[:, 1],
        view.uniform_sky_profile.evaluate(separations * (500 / 400)),
        rtol=1e-14,
    )
    record = view.curves()
    assert record.uniform_sky.frame == record.frame != view.uniform_sky_profile.frame
    np.testing.assert_allclose(
        record.uniform_sky.evaluate(separations),
        view.uniform_sky_response(separations),
        rtol=1e-14,
    )
    np.testing.assert_allclose(stellar_intensity_curves(view).separation_lod[1], 0.1)
    np.testing.assert_allclose(
        view.stellar_map(None, 0.0).data,
        host.stellar_intens(0.0 * lod).data,
        rtol=1e-14,
    )
    sky_grid = PixelGrid.regular((4, 4), 0.4, "lod", frame=view.sky_trans.grid.frame)
    np.testing.assert_allclose(
        view.uniform_sky_map(500.0, grid=sky_grid).data,
        resample(host.sky_trans, sky_grid).data,
        atol=1e-14,
    )
    assert any(
        leaf is view._offaxis_backend.reader.grid.intensity
        for leaf in jax.tree.leaves(view)
    )
    traces = []

    @eqx.filter_jit
    def read(module):
        traces.append(None)
        return module.source_psfs(None, 0.07, -0.04).image.data

    expected = read(view)
    changed = eqx.tree_at(
        lambda m: m._offaxis_backend.reader.grid.intensity,
        view,
        view._offaxis_backend.reader.grid.intensity * 2,
    )
    np.testing.assert_allclose(read(changed), expected * 2, atol=1e-18)
    assert len(traces) == 1


@pytest.mark.parametrize("defect", ["missing", "failed", "declaration", "center"])
def test_quadrant_precedence(tmp_path, defect):
    """Fails if rejected quadrant data can escape through auto or blend."""
    m = write_delivery(tmp_path)
    if defect == "missing":
        m["mirror_checks"] = []
    elif defect == "failed":
        with fits.open(tmp_path / "mirror.fits", mode="update") as hdus:
            hdus[0].data *= 2
        for record in m["mirror_checks"]:
            record["psf_sha256"] = file_sha256(tmp_path / "mirror.fits")
    elif defect == "declaration":
        m["reflection"]["x_mirror"] = False
    else:
        m["detector_center_xy"] = [2.0, 2.0]
    save_manifest(tmp_path, m)
    expected = {
        "missing": Reason.SYMMETRY_CHECK_MISSING,
        "failed": Reason.SYMMETRY_CHECK_FAILED,
        "declaration": Reason.SYMMETRY_UNSUPPORTED,
    }
    if defect == "center":
        with pytest.raises(ValueError, match=r"Conflicting|midpoint") as caught:
            route_interpolation(tmp_path, "auto", None, None)
        assert not isinstance(caught.value, sc.AdmissionRefused)
    else:
        with jax.enable_x64(), pytest.raises(sc.AdmissionRefused) as caught:
            route_interpolation(tmp_path, "auto", None, None)
        assert caught.value.reason_bits & int(expected[defect])
        assert not caught.value.reason_bits & int(
            Reason.PRECISION_UNAVAILABLE | Reason.FALLBACK_USED
        )
    with pytest.raises(ValueError, match="midpoint-aware"):
        route_interpolation(tmp_path, "blend", None, None)


def test_independent_checks(admitted, tmp_path):
    """Fails if independent images join training or empirical units are lost."""
    write_delivery(tmp_path)
    _, backend, info = route_interpolation(tmp_path, "sinc", None, None)
    assert backend.reader.grid.intensity.shape[1:3] == (18, 18)
    assert len(info.receipt.empirical_absolute) == 1
    result = backend.create_psf_with_info(0.07, -0.04)
    truth = fits.getdata(tmp_path / "checks.fits")[0]
    np.testing.assert_allclose(
        result.empirical_max_abs,
        np.max(np.abs(np.asarray(result.intensity) - truth)),
        rtol=1e-12,
        # Admission and compiled contraction can differ by a few pixel ULPs.
        atol=8 * np.spacing(np.float64(truth.max())),
    )


def test_auto_fallback_and_bounded(tmp_path):
    """Fails if auto ignores its allowlist, mixes branches or evades a budget."""
    m = write_delivery(tmp_path, full=True)
    _, backend, info = route_interpolation(tmp_path, "auto", None, None)
    assert info.effective == "sinc"
    result = backend.create_psfs_with_info(jnp.array([0.07, 0.2, np.nan]), jnp.zeros(3))
    np.testing.assert_array_equal(result.selected_backend, [1, 0, 1])
    np.testing.assert_array_equal(result.accepted, [True, True, False])
    assert int(result.reason_bits[1]) & int(Reason.OUTSIDE_DOMAIN)
    small_memory = SincOptions(memory=sc.MemoryBudget(1, 1, 1, 1))
    _, _, memory_info = route_interpolation(tmp_path, "auto", small_memory, None)
    assert memory_info.reasons == ("MEMORY_LIMIT",)
    with pytest.raises(sc.MemoryLimit):
        route_interpolation(tmp_path, "sinc", small_memory, None)
    m["support_extent_D"] = [2.0, 2.0]
    save_manifest(tmp_path, m)
    _, fallback, info = route_interpolation(tmp_path, "auto", None, None)
    assert info.effective == "blend"
    assert "NYQUIST_FAIL" in info.reasons
    assert fallback.create_psf_with_info(0.0, 0.0).status == Status.FALLBACK
    with pytest.raises(sc.AdmissionRefused):
        route_interpolation(tmp_path, "sinc", None, None)
    m["support_extent_D"] = [1.0, 1.0]
    save_manifest(tmp_path, m)
    _, bounded, _ = route_interpolation(
        tmp_path,
        "auto",
        SincOptions(policy=sc.ReadPolicy("bounded", jnp.asarray(1e-5))),
        None,
    )
    result = bounded.create_psf_with_info(0.2, 0.0)
    assert not result.accepted
    assert result.selected_backend == 1
    with jax.enable_x64(False):
        _, precision_fallback, info = route_interpolation(tmp_path, "auto", None, None)
        assert info.reasons == ("PRECISION_UNAVAILABLE",)
        assert precision_fallback.create_psf_with_info(0.0, 0.0).accepted
        with pytest.raises(sc.PrecisionUnavailable):
            route_interpolation(tmp_path, "sinc", None, None)


def test_table_consumer(admitted):
    """Fails if the table consumer rescales densities or resamples off its frame.

    The stellar truth is the SciPy footprint integral of the stored samples,
    with the pixel geometry typed from the grids (``footprint_truth``),
    compared to rounding on every pixel. The four-sample table has no pixel
    two cells inside its edge, so every pixel lies where the spline depends
    on its end condition; the truth implements the documented restriction of
    the spline to the table footprint, and the comparison guards scale and
    frame, not the edge physics.
    """
    host = _YIPData(admitted.package, interpolation="sinc")
    # The record stores density per (lambda/D)^2: no pitch factor.
    density = host.performance.evaluate("leakage_density_lod2", 0.0, 0.0)
    np.testing.assert_allclose(
        admitted.curves().evaluate("leakage_density_lod2", 0.0, 0.0) / 30.0,
        density / 30.0,
        rtol=0,
    )
    x, y = jnp.array([0.07]), jnp.array([-0.04])
    for pitch in (0.25, 0.4):
        stellar = admitted.stellar_map(None, 0.0)
        grid = PixelGrid.focal(4, pitch, stellar.grid.frame)
        assert_stellar_read(
            admitted.stellar_map(500.0, 0.0, grid=grid).data,
            stellar.integrated().data,
            stellar.grid,
            grid,
        )
        grid = PixelGrid.focal(4, pitch, admitted.sky_trans.grid.frame)
        np.testing.assert_array_equal(
            admitted.uniform_sky_map(500.0, grid=grid).data,
            resample(admitted.sky_trans, grid).data,
        )
        grid = PixelGrid.focal(4, pitch, admitted.grid.frame)
        expected = resample(
            admitted._offaxis_backend.create_psfs(x, y).integrated(), grid
        ).data
        result = admitted.source_psfs(500.0, x, y, grid=grid).image.data
        np.testing.assert_allclose(result, expected, atol=1e-18)
        np.testing.assert_array_equal(
            admitted.source_psfs(600.0, x, y, grid=grid).image.data, result
        )


@pytest.mark.medium(reason="builds and reloads cache variants and JITs a scene render")
def test_cache_identity_and_selections(admitted, monkeypatch):
    """Fails if route provenance collides or cube cache loses refusal selections."""
    host = _YIPData(admitted.package, interpolation="sinc")
    original = host.interpolation_info
    base_cube = host._datacube_cache_path
    base_perf = host._perf_filename()
    for key in (
        "manifest_sha256",
        "product_id",
        "requested",
        "effective",
        "yippy_version",
        "kernel_profile",
        "receipt_id",
    ):
        changed = copy.copy(host)
        changed.interpolation_info = replace(original, **{key: "changed"})
        assert changed._datacube_cache_path != base_cube
        assert changed._perf_filename() != base_perf
        assert not changed._datacube_cache_path.exists()
        assert not (changed._perf_dir / changed._perf_filename()).exists()
        assert cache_identity_digest(
            changed._auto_perf_identity()
        ) != cache_identity_digest(host._auto_perf_identity())
        changed._load_or_compute_performance()
        assert (changed._perf_dir / changed._perf_filename()).exists()
        changed._create_delivery_datacube(4)
        first = np.asarray(changed.psf_datacube)

        def no_reads(*args, **kwargs):
            raise AssertionError("cube reload performed source reads")

        with monkeypatch.context() as patch:
            patch.setattr(changed, "create_psfs_with_info", no_reads)
            changed._create_delivery_datacube(4)
        np.testing.assert_array_equal(changed.psf_datacube, first)
        perf_path = changed._perf_dir / changed._perf_filename()
        stamp = perf_path.stat().st_mtime_ns
        changed._load_or_compute_performance()
        assert perf_path.stat().st_mtime_ns == stamp
    host._create_delivery_datacube(4)
    before = np.asarray(host.psf_datacube)
    selected = host.psf_datacube_info["selected_backend"].copy()
    host._create_delivery_datacube(4)
    np.testing.assert_array_equal(host.psf_datacube, before)
    np.testing.assert_array_equal(host.psf_datacube_info["selected_backend"], selected)
    np.testing.assert_array_equal(
        host.psf_datacube_info["accepted"], [[True, False], [True, False]]
    )
    for row, y in enumerate((0.125, 0.375)):
        np.testing.assert_allclose(
            before[row, 0],
            host.offax.create_psf(0.125, y).integrated().data,
            rtol=1e-14,
        )
        assert np.isfinite(before[row, 0]).all()
        assert np.isnan(before[row, 1]).all()
    base_cube.with_suffix(".status.npz").unlink()
    host._create_delivery_datacube(4)
    assert base_cube.with_suffix(".status.npz").exists()

    scene_model = admitted.with_scene_operator()
    # The per-source route reports its cube and refuses a budget below it.
    cube = scene_model.operators[0].psfs
    assert admitted.scene_operator_bytes()["resident"] == cube.nbytes
    with pytest.raises(MemoryLimit, match="max_bytes"):
        admitted.with_scene_operator(max_bytes=1)
    image = Image(
        jnp.zeros(admitted.grid.shape), admitted.grid, (), Measure.integral(), "flux"
    )
    supported = replace(image, data=image.data.at[2, 2].set(1.0))
    good = scene_model.render_scene(supported, None, grid=admitted.grid)
    assert good.valid and jnp.isfinite(good.image.data).all()
    expected = admitted.source_psfs(None, 0.125, 0.125, grid=admitted.grid)
    np.testing.assert_allclose(good.image.data, expected.image.data[0], rtol=1e-12)
    # Negative y lies outside the asymmetric domain even where positive y is supported.
    unsupported = replace(supported, data=supported.data.at[0, 2].set(1.0))
    bad = eqx.filter_jit(scene_model.render_scene)(
        unsupported, None, grid=admitted.grid
    )
    # The refused source is flagged and adds nothing: the image is the good one.
    assert not bad.valid
    np.testing.assert_allclose(bad.image.data, good.image.data, rtol=0, atol=1e-15)


@pytest.mark.medium(
    reason="admits one coefficient table and compiles its raster render"
)
def test_prefiltered_scene_uses_coefficient_renderer(tmp_path):
    """One lit pixel reproduces a source read, including photon conversion.

    Binary64 accumulation differs by at most 1e-12 of the peak from the
    differently ordered coefficient sum in a single read.
    """
    metadata = write_delivery(tmp_path)
    metadata["support_extent_D"] = [0.8, 0.8]
    metadata["served"]["bounds_lod"] = [[-0.2, 0.2], [-0.2, 0.5]]
    metadata["sample_kind"] = "mean"
    metadata["checks"] = []
    save_manifest(tmp_path, metadata)
    with jax.enable_x64():
        prepared = open_yip(tmp_path).prepare(interpolation="sinc")
        # Every route is admitted by estimate, the prefiltered one included.
        with pytest.raises(MemoryLimit, match="max_bytes"):
            prepared.with_scene_operator(max_bytes=1)
        model = prepared.with_scene_operator()
        assert isinstance(model.operators[0], sc.SceneRenderer)
        assert isinstance(model.operators[0], AbstractSceneOperator)
        flux = jnp.zeros(model.grid.shape).at[2, 1].set(2.0)
        scene = Image(flux, model.grid, (), Measure.integral(), "flux")
        result = eqx.filter_jit(model.render_scene)(scene, None, grid=model.grid)
        point = model.source_psfs(None, -0.125, 0.125)
        assert result.valid and point.valid[0]
        expected = 2.0 * point.image.data[0]
        np.testing.assert_allclose(
            result.image.data, expected, rtol=0, atol=1e-12 * expected.max()
        )
        x, y = model.grid.coordinates()
        raster = model.operators[0].render_raster(flux, x[0], y[:, 0])
        direct = raster.intensity * model._offaxis_backend.conversion
        data, covered = model.operators[0].render(flux, model.grid)
        assert covered[2, 1] and bool(raster.accepted)
        np.testing.assert_array_equal(data, direct)
        assert not covered[0, 0] and bool(jnp.all(jnp.isfinite(data)))
        lit = replace(scene, data=flux.at[0, 0].set(1.0))
        assert not model.render_scene(lit, None, grid=model.grid).valid

        bounded = (
            open_yip(tmp_path)
            .prepare(
                interpolation="sinc",
                sinc_options=SincOptions(
                    policy=sc.ReadPolicy("bounded", jnp.asarray(1.0))
                ),
            )
            .with_scene_operator()
        )
        assert isinstance(bounded.operators[0], sc.SceneRenderer)
        dark = replace(scene, data=jnp.zeros_like(flux))
        result = bounded.render_scene(dark, None, grid=bounded.grid)
        assert result.valid
        np.testing.assert_array_equal(result.image.data, 0.0)
        assert not bounded.render_scene(scene, None, grid=bounded.grid).valid


@pytest.mark.slow
def test_public_delivery(tmp_path):
    """Fails if independent images disagree by over 1e-5 of their local peaks."""
    configured = os.environ.get("YIPPY_SINC_DELIVERY")
    if not configured or not (Path(configured) / "sinc-delivery.json").exists():
        pytest.skip("Set YIPPY_SINC_DELIVERY to an independent intensity delivery")
    with jax.enable_x64():
        root = tmp_path / "delivery"
        shutil.copytree(configured, root, ignore=shutil.ignore_patterns("yippy_cache"))
        coro = open_yip(root).prepare(interpolation="sinc")
        host = _YIPData(open_yip(root), interpolation="sinc")
        np.testing.assert_allclose(
            coro.source_psfs(None, 4.0, 5.5).image.data[0],
            host.offax.create_psf(4.0, 5.5).data,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            coro.stellar_map(None, 0.0).data,
            host.stellar_intens(0.0 * lod).data,
            rtol=1e-14,
        )
        assert coro.stellar_grid.frame != coro.grid.frame
        np.testing.assert_allclose(
            coro.uniform_sky_map(500.0, grid=coro.sky_trans.grid).data,
            resample(host.sky_trans, coro.sky_trans.grid).data,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            coro.stellar_map(500.0, 0.0, grid=coro.stellar_grid).data,
            resample(
                host.stellar_intens(0.0 * lod).integrated(), coro.stellar_grid
            ).data,
            atol=1e-14,
        )
        x, y = jnp.array([4.0]), jnp.array([5.5])
        np.testing.assert_allclose(
            coro.source_psfs(500.0, x, y, grid=coro.grid).image.data,
            resample(host.offax.create_psfs(x, y).integrated(), coro.grid).data,
            rtol=1e-14,
        )
        metadata = json.loads((root / "sinc-delivery.json").read_text())
        coords = np.genfromtxt(root / "starter_checks.csv", names=True, delimiter=",")
        points = np.column_stack((coords["x_lod"], coords["y_lod"]))
        path = root / "checks" / "mono-000-float64.fits"
        if not path.exists():
            path = root / "checks" / "mono-000.fits"
        truth = fits.getdata(path)
        assert len(points) == 16
        pitch = np.array(metadata["pixel_scale_lod_xy"])
        center = np.array(metadata["detector_center_xy"])
        yy, xx = np.indices(truth.shape[-2:])
        for q, image in zip(points, truth, strict=True):
            region = ((xx - center[0]) * pitch[0] - q[0]) ** 2 + (
                (yy - center[1]) * pitch[1] - q[1]
            ) ** 2 <= 1.0
            peak = image[region].max()
            assert peak > 0
            assert (
                np.max(
                    np.abs(
                        np.asarray(
                            coro.source_psfs(None, *q).image.integrated().data[0]
                        )
                        - image
                    )
                )
                <= 1e-5 * peak
            )


def test_bounded_legacy_refuses(tmp_path):
    """A bounded request cannot construct an unbounded legacy fallback."""
    legacy = write_synthetic_yip(
        tmp_path / "legacy", npix=4, max_offset_lod=1.0, offset_step_lod=0.5
    )
    with (
        jax.enable_x64(),
        pytest.raises(ValueError, match=r"[Bb]ounded"),
    ):
        open_yip(legacy.path).prepare(
            interpolation="auto",
            sinc_options=SincOptions(
                policy=sc.ReadPolicy("bounded", jnp.asarray(1e-20))
            ),
        )


@pytest.mark.parametrize(
    "unit", ["deg", "rad", "arcsec", "mas", "lambda/D", "lod", "", "absent"]
)
def test_header_explicit_angular_units(tmp_path, unit):
    """Equivalent angular header pitches agree after explicit conversion."""
    m = write_delivery(tmp_path, full=True)
    radians = 0.25 * (500e-9 / 6.0)
    values = {
        "deg": radians * (180 / np.pi),
        "rad": radians,
        "arcsec": m["pixel_scale_arcsec_xy"][0],
        "mas": m["pixel_scale_arcsec_xy"][0] * 1000,
        "lambda/D": 0.25,
        "lod": 0.25,
        "": 0.25,
        "absent": 0.25,
    }
    with fits.open(tmp_path / "nodes.fits", mode="update") as hdus:
        del hdus[0].header["PIXSCALE"]
        if unit == "absent":
            hdus[0].header["PIXSCALE"] = values[unit]
        else:
            hdus[0].header["PIXSCALE"] = (values[unit], unit + "/pixel" if unit else "")
    m["planes"][0]["psf_sha256"] = file_sha256(tmp_path / "nodes.fits")
    save_manifest(tmp_path, m)
    assert load_delivery(tmp_path, m, SincOptions()).header.pixscale.value == 0.25


@pytest.mark.parametrize("unit", ["deg/pixel", "furlongs"])
def test_header_invalid_angular_units(tmp_path, unit):
    """Fails if conflicting degrees or unknown nonempty units are accepted."""
    m = write_delivery(tmp_path, full=True)
    with fits.open(tmp_path / "nodes.fits", mode="update") as hdus:
        hdus[0].header["PIXSCALE"] = (0.25, unit)
    m["planes"][0]["psf_sha256"] = file_sha256(tmp_path / "nodes.fits")
    save_manifest(tmp_path, m)
    with pytest.raises(ValueError, match=r"PIXSCALE|unit"):
        route_interpolation(tmp_path, "auto", None, None)


@pytest.mark.parametrize("gate", ["precision", "memory"])
@pytest.mark.parametrize(
    "defect",
    [
        "check_hash",
        "domain",
        "reflection",
        "mirror_hash",
        "index",
        "psf_hash",
        "header",
        "duplicate",
        "hole",
        "row",
        "coordinates",
    ],
)
def test_validation_precedes_fallback(tmp_path, gate, defect):
    """Resource routing cannot hide corrupt declarations or independent files."""
    m = write_delivery(tmp_path, full=True)
    if defect == "psf_hash":
        m["planes"][0]["psf_sha256"] = "0" * 64
    elif defect == "header":
        with fits.open(tmp_path / "nodes.fits", mode="update") as hdus:
            hdus[0].header["PIXSCALE"] = (0.2, "lambda/D/pixel")
        m["planes"][0]["psf_sha256"] = file_sha256(tmp_path / "nodes.fits")
    elif defect in ("duplicate", "hole", "row", "coordinates"):
        path = tmp_path / "training.csv"
        rows = path.read_text().splitlines()
        final = rows[-1].split(",")
        if defect == "duplicate":
            final[1:3] = rows[-2].split(",")[1:3]
        elif defect == "hole":
            final[1] = "18"
        elif defect == "row":
            final[0] = "0"
        else:
            final[3] = "99.0"
        rows[-1] = ",".join(final)
        path.write_text("\n".join(rows) + "\n")
    elif defect == "check_hash":
        m["checks"][0]["psf_sha256"] = "0" * 64
    elif defect == "domain":
        m["served"]["bounds_lod"][0] = [0.1, -0.1]
    elif defect == "reflection":
        m["reflection"] = {"kind": "bad"}
    elif defect == "mirror_hash":
        m["mirror_checks"] = [dict(m["checks"][0], base_index=0, psf_sha256="0" * 64)]
    else:
        m["checks"][0]["image_index"] = -1
    save_manifest(tmp_path, m)
    options = (
        SincOptions(memory=sc.MemoryBudget(1, 1, 1, 1)) if gate == "memory" else None
    )
    with jax.enable_x64(gate != "precision"), pytest.raises(ValueError):
        route_interpolation(tmp_path, "auto", options, None)


@pytest.mark.parametrize(
    "field",
    [
        "coordinate_model",
        "diameter_convention",
        "optical_state",
        "premises",
        "support_extent_D",
        "support_evidence",
        "sampling_min_wavelength_m",
        "served",
    ],
)
def test_missing_admission_metadata(tmp_path, field):
    """Absent physical claims route auto with METADATA_MISSING, never KeyError."""
    m = write_delivery(tmp_path, full=True)
    del (m["planes"][0] if field == "sampling_min_wavelength_m" else m)[field]
    save_manifest(tmp_path, m)
    with jax.enable_x64():
        _, backend, info = route_interpolation(tmp_path, "auto", None, None)
        assert info.reasons == ("METADATA_MISSING",)
        assert backend.create_psf_with_info(0.0, 0.0).accepted
        with pytest.raises(sc.AdmissionRefused) as caught:
            route_interpolation(tmp_path, "sinc", None, None)
        assert caught.value.reason_bits == int(Reason.METADATA_MISSING)


def test_campaign_claim_refuses(tmp_path):
    """A failed explicit campaign claim needs a revised manifest to proceed."""
    m = write_delivery(tmp_path, full=True)
    m["spectral_sampling"]["campaign_sampling_min_m"] = 450e-9
    save_manifest(tmp_path, m)
    with jax.enable_x64():
        with pytest.raises(ValueError, match=r"campaign.*revised manifest"):
            route_interpolation(tmp_path, "auto", None, None)
        m["spectral_sampling"]["campaign_sampling_min_m"] = None
        save_manifest(tmp_path, m)
        assert route_interpolation(tmp_path, "auto", None, None)[2].effective == "sinc"


@pytest.mark.parametrize("gate", ["sampling", "precision"])
def test_fallback_options_cache(tmp_path, monkeypatch, gate):
    """Distinct fallback domains/options generate and reload separate artifacts."""
    m = write_delivery(tmp_path, full=True)
    if gate == "sampling":
        m["support_extent_D"] = [2.0, 2.0]
    save_manifest(tmp_path, m)
    with jax.enable_x64():
        domains = [
            sc.Domain("box", "closed", jnp.array([[-r, -r], [r, r]]), jnp.zeros(2))
            for r in (0.09, 0.08)
        ]
        requests = [SincOptions(domain=d) for d in domains]
        requests.extend(
            [
                SincOptions(domain=domains[0], product_id="mono"),
                SincOptions(
                    domain=domains[0],
                    memory=sc.MemoryBudget(10000000, 10000000, 10000000, 10000000),
                ),
            ]
        )
        with jax.enable_x64(gate != "precision"):
            objects = [
                _YIPData(open_yip(tmp_path), interpolation="auto", sinc_options=o)
                for o in requests
            ]
            assert all(c.interpolation_info.effective == "blend" for c in objects)
            assert len({c._datacube_cache_path for c in objects}) == len(objects)
            assert len({c._perf_filename() for c in objects}) == len(objects)
            for i, coro in enumerate(objects):
                entry = reload = coro._create_delivery_datacube
                assert not coro._datacube_cache_path.exists()
                entry(4)
                before = np.asarray(coro.psf_datacube)
                assert np.isfinite(before).all()
                assert (coro._perf_dir / coro._perf_filename()).exists()

                def no_reads(*args, **kwargs):
                    raise AssertionError("cache reload performed source reads")

                with monkeypatch.context() as patch:
                    patch.setattr(coro, "create_psfs_with_info", no_reads)
                    reload(4)
                np.testing.assert_array_equal(coro.psf_datacube, before)
                # Reconstructing the same request also reuses its performance entry.
                perf_path = coro._perf_dir / coro._perf_filename()
                stamp = perf_path.stat().st_mtime_ns
                again = _YIPData(
                    open_yip(tmp_path), interpolation="auto", sinc_options=requests[i]
                )
                assert again._perf_filename() == coro._perf_filename()
                assert again._datacube_cache_path == coro._datacube_cache_path
                assert perf_path.stat().st_mtime_ns == stamp


@pytest.mark.parametrize(
    "measure,factor", [("point", 2.0), ("integral", 2.0), ("mean", 0.125)]
)
def test_intake_finite_error_units(tmp_path, measure, factor):
    """Real intake scales nonzero finite empirical residuals with pixel values."""
    m = write_delivery(tmp_path)
    with jax.enable_x64():
        _, reference, _ = route_interpolation(tmp_path, "sinc", None, None)
        raw = reference.create_psf_with_info(0.07, -0.04)
        assert np.isfinite(raw.empirical_max_abs) and raw.empirical_max_abs > 0
        m["sample_kind"] = measure
        m["normalization"]["kind"] = "unocculted_peak"
        m["normalization"]["denominator_definition"] = (
            "Full incident reference intensity"
        )
        m["planes"][0]["photon_conversion"] = 2.0
        save_manifest(tmp_path, m)
        _, backend, info = route_interpolation(tmp_path, "sinc", None, None)
        converted = backend.create_psf_with_info(0.07, -0.04)
        image = backend.create_psf(0.07, -0.04)
        expected_measure = (
            Measure.density("grid") if measure == "point" else Measure.integral()
        )
        assert image.measure == expected_measure == backend.reader.grid.products.measure
        np.testing.assert_allclose(
            image.integrated().data, raw.intensity * factor, rtol=1e-14
        )
        np.testing.assert_allclose(
            converted.intensity, raw.intensity * factor, rtol=1e-14
        )
        np.testing.assert_allclose(
            converted.empirical_max_abs, raw.empirical_max_abs * factor, rtol=1e-12
        )
        retained = dict(info.receipt.provenance)["manifest"]
        assert json.loads(retained) == m
        assert info.receipt.offset_dtype == "float64"
        assert all(info.receipt.axis_exact)


def test_spatial_cross_frame_aperture():
    """The stellar aperture maps pitch and wavelength before integrating density."""
    from optixstuff.performance import circular_aperture
    from scipy.special import j1

    with jax.enable_x64():
        n, radius, level = 12, 2.0, 1e-10
        _, xx = np.indices((n, n))
        center = (n - 1) / 2
        star = level * (1 + 0.4 * np.cos(2 * np.pi * (xx - center) / n))
        ratio = (0.25 * 500) / (0.4 * 400)
        k = 2 * np.pi * ratio / n
        expected = (
            np.pi
            * radius**2
            * level
            * ratio**2
            * (1 + 0.4 * 2 * j1(k * radius) / (k * radius))
        )

        def integrate(scale):
            return circular_aperture(
                jnp.ones((16, 16)),
                radius,
                0.25,
                oversample=40,
                half_width_pix=4,
                fixed_center=(7.5, 7.5),
                star_psf=jnp.array(star),
                star_frame=(center, center, 7.5, 7.5, scale),
            )[2]

        assert abs(float(integrate(ratio)) / expected - 1) < 5e-6
        assert abs(float(integrate(0.25 / 0.4)) / expected - 1) > 0.1


@pytest.mark.parametrize("budget", [1e-5, 1e-6])
def test_policy_cache_generation(admitted, budget):
    """Bounded policies identify generated cubes and reloaded performance separately."""
    host = _YIPData(admitted.package, interpolation="sinc")
    changed = copy.copy(host)
    changed.offax = eqx.tree_at(
        lambda b: b.reader.policy,
        host.offax,
        sc.ReadPolicy("bounded", jnp.asarray(budget)),
    )
    assert changed._datacube_cache_path != host._datacube_cache_path
    assert changed._perf_filename() != host._perf_filename()
    changed._load_or_compute_performance()
    changed._create_delivery_datacube(4)
    before = np.asarray(changed.psf_datacube)
    changed._create_delivery_datacube(4)
    np.testing.assert_array_equal(changed.psf_datacube, before)
    assert not changed.psf_datacube_info["accepted"].any()
    assert changed._datacube_cache_path.exists()
    assert (changed._perf_dir / changed._perf_filename()).exists()


def test_original_offset_precision(tmp_path):
    """The core receipt retains the delivered float32 offset precision."""
    m = write_delivery(tmp_path)
    path = tmp_path / "offsets.fits"
    values = fits.getdata(path).astype(np.float32)
    fits.writeto(path, values, overwrite=True)
    m["planes"][0]["offsets_sha256"] = file_sha256(path)
    save_manifest(tmp_path, m)
    with jax.enable_x64():
        _, backend, info = route_interpolation(tmp_path, "sinc", None, None)
        assert info.receipt.offset_dtype == "float32"
        assert info.receipt.offset_tolerance_xy[0] > 1e-6
        assert backend.create_psf_with_info(0.07, -0.04).accepted


@pytest.mark.parametrize(
    "gate", ["normal", "blend", "precision", "memory", "metadata", "sampling"]
)
@pytest.mark.parametrize(
    "defect",
    [
        "normalization",
        "support",
        "external_evidence",
        "peak_zero",
        "peak_negative",
        "peak_inf",
        "peak_nan",
        "entrance_tilt_model",
        "fixed_linear_optics",
        "fixed_detector_registration",
        "fixed_pixel_response",
        "source_independent_normalization",
        "source_independent_spectral_weights",
        "false_with_missing_support",
        "wavelength_inf",
        "wavelength_overflow",
        "negative_subnormal",
        "empty_product_id",
        "band_edges",
        "check_domain",
        "check_stencil",
    ],
)
def test_admission_declaration_errors(tmp_path, gate, defect):
    """Fails if a fallback masks hard invalid evidence or a physical conflict."""
    m = write_delivery(tmp_path, full=True)
    if defect == "normalization":
        m["normalization"]["kind"] = "unknown_reference"
    elif defect == "support":
        m["support_extent_D"] = [1.0]
    elif defect == "external_evidence":
        m["premises"]["fixed_linear_optics"]["evidence"] = 2
    elif defect.startswith("peak_"):
        m["checks"][0]["local_peak_reference"] = {
            "peak_zero": 0.0,
            "peak_negative": -1.0,
            "peak_inf": np.inf,
            "peak_nan": np.nan,
        }[defect]
    elif defect == "false_with_missing_support":
        m["premises"]["source_independent_spectral_weights"]["value"] = False
        del m["support_evidence"]
    elif defect in ("wavelength_inf", "wavelength_overflow"):
        wavelength = np.inf if defect == "wavelength_inf" else 1e308
        m["planes"][0]["wavelength_m"] = wavelength
        m["planes"][0]["shortest_contributing_m"] = wavelength
        m["planes"][0]["band_edges_m"] = [wavelength, wavelength]
        m["spectral_sampling"]["shortest_contributing_m"] = wavelength
        if defect == "wavelength_overflow":
            with fits.open(tmp_path / "nodes.fits", mode="update") as hdus:
                hdus[0].header["MINLAM"] = (wavelength, "m")
                hdus[0].header["MAXLAM"] = (wavelength, "m")
            m["planes"][0]["psf_sha256"] = file_sha256(tmp_path / "nodes.fits")
    elif defect == "negative_subnormal":
        with fits.open(tmp_path / "nodes.fits", mode="update") as hdus:
            hdus[0].data.view(">u4")[0, 0, 0] = 0x80000001
        m["planes"][0]["psf_sha256"] = file_sha256(tmp_path / "nodes.fits")
    elif defect == "empty_product_id":
        m["planes"][0]["id"] = ""
        m["checks"][0]["plane_id"] = ""
    elif defect == "band_edges":
        m["planes"][0]["band_edges_m"] = [400e-9, 600e-9]
    elif defect == "check_domain":
        m["checks"][0]["source_lod_xy"] = [0.2, 0.0]
    elif defect == "check_stencil":
        m["checks"][0]["source_lod_xy"] = [0.5, 0.0]
        m["served"]["bounds_lod"] = [[-0.6, 0.6], [-0.6, 0.6]]
    else:
        m["premises"][defect]["value"] = False
    if gate == "metadata":
        del m["coordinate_model"]
    elif gate == "sampling" and defect != "support":
        m["support_extent_D"] = [2.0, 2.0]
    save_manifest(tmp_path, m)
    options = (
        SincOptions(memory=sc.MemoryBudget(1, 1, 1, 1)) if gate == "memory" else None
    )
    with jax.enable_x64(gate != "precision"), pytest.raises(ValueError):
        route_interpolation(
            tmp_path, "blend" if gate == "blend" else "auto", options, None
        )


@pytest.mark.medium
@pytest.mark.parametrize("peak", ["absent", None, 0.001])
def test_peak_reference_controls(tmp_path, peak):
    """Fails if unknown or positive peak evidence prevents admission or fallback.

    Cold independent-check admission and finite reads cover all six routes.
    """
    m = write_delivery(tmp_path, full=True)
    if peak == "absent":
        del m["checks"][0]["local_peak_reference"]
    else:
        m["checks"][0]["local_peak_reference"] = peak
    for gate in (
        "normal",
        "blend",
        "precision",
        "memory",
        "metadata",
        "sampling",
    ):
        metadata = copy.deepcopy(m)
        if gate == "metadata":
            del metadata["coordinate_model"]
        elif gate == "sampling":
            metadata["support_extent_D"] = [2.0, 2.0]
        save_manifest(tmp_path, metadata)
        options = (
            SincOptions(memory=sc.MemoryBudget(1, 1, 1, 1))
            if gate == "memory"
            else None
        )
        with jax.enable_x64(gate != "precision"):
            _, backend, info = route_interpolation(
                tmp_path, "blend" if gate == "blend" else "auto", options, None
            )
            assert info.effective == ("sinc" if gate == "normal" else "blend")
            result = backend.create_psf_with_info(0.0, 0.0)
            assert result.accepted and np.isfinite(result.intensity).all()


def test_override_domain_excludes_unreadable_checks(tmp_path):
    """A narrower caller domain excludes, rather than rejects, delivered checks."""
    m = write_delivery(tmp_path, full=True)
    save_manifest(tmp_path, m)
    check_ids = [r["id"] for r in m["checks"]]
    with jax.enable_x64():
        narrow = sc.Domain(
            "box", "closed", jnp.array([[-0.05, -0.05], [0.05, 0.05]]), jnp.zeros(2)
        )
        _, backend, info = route_interpolation(
            tmp_path, "sinc", SincOptions(domain=narrow), None
        )
        provenance = dict(info.receipt.provenance)
        excluded = json.loads(provenance["excluded_checks"])
        assert excluded and set(excluded) <= set(check_ids)
        result = backend.create_psf_with_info(0.0, 0.0)
        assert bool(result.accepted)


@pytest.mark.parametrize("kind", ["integral", "mean"])
def test_measured_detector_cells_survive_explicit_native_grid(tmp_path, kind):
    """Fails if a native read applies the declared detector response twice."""
    m = write_delivery(tmp_path, full=True)
    m["sample_kind"] = kind
    save_manifest(tmp_path, m)
    with jax.enable_x64():
        model = open_yip(tmp_path).prepare(interpolation="blend")
        source = model.source_psfs(None, 0.0, 0.0, grid=model.grid).image
        native = model.source_psfs(None, 0.0, 0.0).image
        assert np.ptp(native.data) > 0
        assert native.measure == source.measure == Measure.integral()
        assert native.reconstruction == "nearest"
        np.testing.assert_array_equal(native.data, source.data)
        stored = model.package.offaxis_psfs[16 * 33 + 16]
        expected = stored if kind == "integral" else stored * 0.25**2
        np.testing.assert_allclose(source.data[0], expected, rtol=2e-7)
