"""Optional off-axis routing, geometry and refusal regression tests."""

import copy
import importlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from lod_unit import lod
from sinc_fixture import save_manifest, write_delivery
from synthetic_yip import write_synthetic_yip

from yippy import Coronagraph, EqxCoronagraph, SincOptions
from yippy.sinc_delivery import detect_delivery, file_sha256, load_delivery
from yippy.sinc_routing import SincExtraMissing, route_interpolation
from yippy.util import cache_identity_digest


@pytest.fixture(scope="module")
def admitted(tmp_path_factory):
    """One tiny admitted quadrant and real performance cache for all reads."""
    pytest.importorskip("sinchole")
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
            patch.setattr("yippy.coronagraph.OffJAX", reject_legacy)
            coro = Coronagraph(root, interpolation="sinc")
        yield coro
    jax.clear_caches()


def test_default_and_missing_extra(tmp_path, monkeypatch):
    """Fails if default blend imports sinc or missing-extra auto changes images."""
    synthetic_1d = write_synthetic_yip(
        tmp_path / "legacy", npix=4, max_offset_lod=1.0, offset_step_lod=0.5
    )
    original = importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name == "sinchole":
            raise ModuleNotFoundError("no sinchole", name="sinchole")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    blend = Coronagraph(synthetic_1d.path)
    auto = Coronagraph(synthetic_1d.path, interpolation="auto")
    np.testing.assert_array_equal(blend.create_psf(0.3, 0.2), auto.create_psf(0.3, 0.2))
    assert not auto.create_psf_with_info(np.nan, 0.0).accepted
    assert np.isnan(auto.create_psf(np.nan, 0.0)).all()
    assert auto.interpolation_info.reasons == ("EXTRA_MISSING",)
    assert "interpolation" not in blend._datacube_identity()
    with pytest.raises(SincExtraMissing, match=r"yippy\[sinc\]"):
        Coronagraph(synthetic_1d.path, interpolation="sinc")

    def broken(name):
        raise ModuleNotFoundError(
            "missing internal dependency", name="internal_dependency"
        )

    monkeypatch.setattr(importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="internal dependency"):
        Coronagraph(synthetic_1d.path, interpolation="auto")


@pytest.mark.parametrize("defect", ["schema", "hash", "header", "duplicate", "hole"])
def test_delivery_corruption(tmp_path, defect):
    """Fails if auto hides malformed declarations or corrupt measured data."""
    pytest.importorskip("sinchole")
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


def test_full_blend_extra_isolation(tmp_path, monkeypatch):
    """Fails if full-grid blend imports sinc or scales mean photon units twice."""
    m = write_delivery(tmp_path, full=True)
    m["normalization"]["kind"] = "unocculted_peak"
    m["planes"][0]["photon_conversion"] = 2.0
    m["sample_kind"] = "mean"
    save_manifest(tmp_path, m)
    original = importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name == "sinchole":
            raise ModuleNotFoundError("no sinchole", name="sinchole")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    delivery, blend, _ = route_interpolation(tmp_path, "blend", None, None)
    _, fallback, info = route_interpolation(tmp_path, "auto", None, None)
    assert info.reasons == ("EXTRA_MISSING",)
    expected = delivery.nodes[16 * 33 + 16] * (2.0 * 0.25**2)
    np.testing.assert_allclose(blend.create_psf(0.0, 0.0), expected, rtol=2e-7)
    np.testing.assert_array_equal(
        fallback.create_psf(0.0, 0.0), blend.create_psf(0.0, 0.0)
    )
    assert fallback.create_psf_with_info(0.0, 0.0).selected_backend == 0


def test_product_selection(tmp_path):
    """Fails if a multi-plane manifest silently selects by list position."""
    m = write_delivery(tmp_path)
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
    chosen = load_delivery(tmp_path, m, SincOptions(product_id="other"))
    assert chosen.header.pixscale.value == 0.25
    np.testing.assert_allclose(chosen.header.lambda0.to_value("nm"), 500.0, atol=1e-12)
    assert chosen.plane["id"] == "other"
    assert chosen.nodes.dtype == np.float32
    bad_check = dict(m["checks"][0], image_index=-1)
    with pytest.raises(ValueError, match="image_index"):
        chosen.read_images([bad_check])


def test_quadrant_native_intake(admitted):
    """Fails if manifest intake applies legacy center, mirrors or float casting."""
    grid = admitted.offax.reader.grid
    assert grid.intensity.dtype == jnp.float32
    assert grid.intensity.shape == (1, 18, 18, 4, 4)
    np.testing.assert_array_equal(grid.products.pixels.center_xy_pix, [1.5, 1.5])


def test_reads_units_refusals(admitted):
    """Fails on integral/mean conversion, clipped sinc output or NaN acceptance."""
    sc = pytest.importorskip("sinchole")
    reader = admitted.offax.reader
    direct = reader.read(jnp.array([0.07, -0.04]), jnp.int32(0))
    actual = admitted.create_psf_with_info(0.07, -0.04)
    np.testing.assert_allclose(actual.intensity, direct.intensity, atol=1e-18)
    assert actual.selected_backend == 1
    assert actual.ledger.target_id.endswith(
        ":" + admitted.interpolation_info.product_id
    )
    assert actual.ledger.target_id.startswith("incident_photon_per_pixel:")
    # Replacing leaves here tests the conversion operator only. Public reuse
    # requires fresh admission and evidence for changed samples.
    signed_backend = eqx.tree_at(
        lambda b: b.reader.grid.intensity,
        admitted.offax,
        -admitted.offax.reader.grid.intensity,
    )
    np.testing.assert_allclose(
        signed_backend.create_psf(0.07, -0.04), -direct.intensity, atol=1e-18
    )
    for measure, conversion in (("integral", 2.0), ("mean", 2.0 * 0.25**2)):
        d = admitted._sinc_delivery
        metadata = copy.deepcopy(d.manifest)
        metadata["sample_kind"] = measure
        metadata["normalization"]["kind"] = "unocculted_peak"
        metadata["planes"][0]["photon_conversion"] = 2.0
        from yippy.delivery_backend import DeliveryBackend
        from yippy.sinc_delivery import admit_delivery

        changed = replace(
            d, manifest=metadata, plane=metadata["planes"][0], conversion=conversion
        )
        admission, domain, memory = admit_delivery(changed, SincOptions(), sc)
        new_reader = sc.Interpolator(
            admission.require(),
            domain,
            sc.KernelProfile(),
            sc.ReadPolicy(),
            sc.Tiling(),
            memory,
        )
        backend = DeliveryBackend(changed, new_reader, "sinc")
        np.testing.assert_allclose(
            backend.create_psf(0.07, -0.04), direct.intensity * conversion, atol=1e-18
        )
    refused = admitted.create_psf_with_info(np.nan, 0.0)
    assert not refused.accepted
    assert int(refused.reason_bits) & int(sc.Reason.NONFINITE_QUERY)
    with pytest.raises(sc.ReadRefused):
        admitted.create_psf(np.nan, 0.0)
    assert np.isnan(jax.jit(admitted.create_psf)(np.nan, 0.0)).all()
    batch = admitted.create_psfs_with_info(
        jnp.array([0.07, np.nan]), jnp.array([-0.04, 0.0])
    )
    np.testing.assert_array_equal(batch.accepted, [True, False])
    assert admitted.create_psfs(jnp.empty(0), jnp.empty(0)).shape == (0, 4, 4)


def test_stellar_frame_and_eqx(admitted):
    """Fails if stellar pitch is overwritten or reader tables hide in closures."""
    view = EqxCoronagraph(yippy_coro=admitted)
    assert np.ptp(view.stellar_intens(0.0)) > 0
    from hwoutils.transforms import resample_flux

    expected = resample_flux(view.stellar_intens(0.0), 0.4, 0.3, (4, 4), 0.0)
    wrong = resample_flux(view.stellar_intens(0.0), 0.25, 0.3, (4, 4), 0.0)
    np.testing.assert_array_equal(
        view.stellar_map(500.0, 0.0, pixel_scale_lod=0.3, shape=(4, 4)), expected
    )
    assert not np.allclose(expected, wrong, atol=0.0, rtol=1e-3)
    assert view.stellar_pixel_scale_lod == 0.4
    assert view.stellar_frame == (400.0, 6.0)
    assert admitted.offaxis_header.lambda0.to_value("nm") == pytest.approx(500.0)
    assert view.pixel_scale_lod == 0.25
    assert admitted.header.pixscale.value == 0.4
    np.testing.assert_allclose(admitted.core_intensity_separations[1], 0.1)
    np.testing.assert_allclose(
        view.stellar_intens(0.0), admitted.stellar_intens(0.0 * lod), rtol=1e-14
    )
    np.testing.assert_allclose(
        view.stellar_map(500.0, 0.0, pixel_scale_lod=0.4, shape=(4, 4)),
        admitted.stellar_intens(0.0 * lod),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        view.background_transmission(500.0, pixel_scale_lod=0.4, shape=(4, 4)),
        admitted.sky_trans(),
        atol=1e-14,
    )
    assert any(
        leaf is admitted.offax.reader.grid.intensity for leaf in jax.tree.leaves(view)
    )

    traces = []

    @eqx.filter_jit
    def read(module):
        traces.append(None)
        return module.create_psf(0.07, -0.04)

    expected = read(view)
    changed = eqx.tree_at(
        lambda m: m._offaxis_backend.reader.grid.intensity,
        view,
        admitted.offax.reader.grid.intensity * 2,
    )
    np.testing.assert_allclose(read(changed), expected * 2, atol=1e-18)
    assert len(traces) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        EqxCoronagraph(yippy_coro=admitted, interpolation="blend")


@pytest.mark.parametrize("defect", ["missing", "failed", "declaration", "center"])
def test_quadrant_precedence(tmp_path, defect):
    """Fails if rejected quadrant data can escape through auto or blend."""
    sc = pytest.importorskip("sinchole")
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
        "missing": sc.Reason.SYMMETRY_CHECK_MISSING,
        "failed": sc.Reason.SYMMETRY_CHECK_FAILED,
        "declaration": sc.Reason.SYMMETRY_UNSUPPORTED,
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
            sc.Reason.PRECISION_UNAVAILABLE | sc.Reason.FALLBACK_USED
        )
    with pytest.raises(ValueError, match="midpoint-aware"):
        route_interpolation(tmp_path, "blend", None, None)


def test_independent_checks(admitted, tmp_path):
    """Fails if independent images join training or empirical units are lost."""
    pytest.importorskip("sinchole")
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
    sc = pytest.importorskip("sinchole")
    m = write_delivery(tmp_path, full=True)
    _, backend, info = route_interpolation(tmp_path, "auto", None, None)
    assert info.effective == "sinc"
    result = backend.create_psfs_with_info(jnp.array([0.07, 0.2, np.nan]), jnp.zeros(3))
    np.testing.assert_array_equal(result.selected_backend, [1, 0, 1])
    np.testing.assert_array_equal(result.accepted, [True, True, False])
    assert int(result.reason_bits[1]) & int(sc.Reason.OUTSIDE_DOMAIN)
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
    assert fallback.create_psf_with_info(0.0, 0.0).status == sc.Status.FALLBACK
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
    """Fails if the table consumer rescales integrals or uses a stellar pitch."""
    pytest.importorskip("optixstuff")
    from optixstuff.yippy_coronagraph import YippyCoronagraph

    view = EqxCoronagraph(yippy_coro=admitted)
    consumer = YippyCoronagraph(backend=view)
    np.testing.assert_allclose(
        consumer.core_mean_intensity(0.0, 500.0),
        view.core_mean_intensity(0.0) / 0.4**2,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        consumer.noise_floor_ayo(0.0),
        (view.core_mean_intensity(0.0) / 30.0) / 0.4**2,
        rtol=1e-14,
    )
    np.testing.assert_array_equal(
        consumer.stellar_map(500.0, 0.0, pixel_scale_lod=0.4, shape=(4, 4)),
        view.stellar_map(500.0, 0.0, pixel_scale_lod=0.4, shape=(4, 4)),
    )
    np.testing.assert_array_equal(
        consumer.background_transmission(500.0, pixel_scale_lod=0.4, shape=(4, 4)),
        view.background_transmission(500.0, pixel_scale_lod=0.4, shape=(4, 4)),
    )
    x, y = jnp.array([0.07]), jnp.array([-0.04])
    expected = admitted.create_psfs(x, y)
    result = consumer.source_psfs(500.0, x, y, pixel_scale_lod=0.25, shape=(4, 4))
    np.testing.assert_allclose(result, expected, atol=1e-18)
    np.testing.assert_array_equal(
        view.source_psfs(600.0, x, y, pixel_scale_lod=0.25, shape=(4, 4)), result
    )


def test_cache_identity_and_selections(admitted, monkeypatch):
    """Fails if route provenance collides or cube cache loses refusal selections."""
    original = admitted.interpolation_info
    base_cube = admitted._datacube_cache_path
    base_perf = admitted._perf_filename()
    for key in (
        "manifest_sha256",
        "product_id",
        "requested",
        "effective",
        "sinchole_version",
        "kernel_profile",
        "receipt_id",
    ):
        changed = copy.copy(admitted)
        changed.interpolation_info = replace(original, **{key: "changed"})
        assert changed._datacube_cache_path != base_cube
        assert changed._perf_filename() != base_perf
        assert not changed._datacube_cache_path.exists()
        assert not (changed._perf_dir / changed._perf_filename()).exists()
        assert cache_identity_digest(
            changed._auto_perf_identity()
        ) != cache_identity_digest(admitted._auto_perf_identity())
        changed._load_or_compute_performance()
        assert (changed._perf_dir / changed._perf_filename()).exists()
        changed.create_psf_datacube(batch_size=4)
        first = np.asarray(changed.psf_datacube)

        def no_reads(*args, **kwargs):
            raise AssertionError("cube reload performed source reads")

        with monkeypatch.context() as patch:
            patch.setattr(changed, "create_psfs_with_info", no_reads)
            changed._create_psf_datacube_gpu(batch_size=4)
        np.testing.assert_array_equal(changed.psf_datacube, first)
        perf_path = changed._perf_dir / changed._perf_filename()
        stamp = perf_path.stat().st_mtime_ns
        changed._load_or_compute_performance()
        assert perf_path.stat().st_mtime_ns == stamp
    admitted.create_psf_datacube(batch_size=4)
    before = np.asarray(admitted.psf_datacube)
    selected = admitted.psf_datacube_info["selected_backend"].copy()
    admitted._create_psf_datacube_gpu(batch_size=4)
    np.testing.assert_array_equal(admitted.psf_datacube, before)
    np.testing.assert_array_equal(
        admitted.psf_datacube_info["selected_backend"], selected
    )
    np.testing.assert_array_equal(
        admitted.psf_datacube_info["accepted"], [[True, False], [True, False]]
    )
    for row, y in enumerate((0.125, 0.375)):
        np.testing.assert_allclose(
            before[row, 0], admitted.create_psf(0.125, y), rtol=1e-14
        )
        assert np.isfinite(before[row, 0]).all()
        assert np.isnan(before[row, 1]).all()
    base_cube.with_suffix(".status.npz").unlink()
    admitted.create_psf_datacube(batch_size=4)
    assert base_cube.with_suffix(".status.npz").exists()


@pytest.mark.slow
def test_public_delivery(tmp_path):
    """Fails if independent images disagree by over 1e-5 of their local peaks."""
    pytest.importorskip("sinchole")
    configured = os.environ.get("YIPPY_SINC_DELIVERY")
    if not configured or not (Path(configured) / "sinc-delivery.json").exists():
        pytest.skip("Set YIPPY_SINC_DELIVERY to an independent intensity delivery")
    with jax.enable_x64():
        root = tmp_path / "delivery"
        shutil.copytree(configured, root, ignore=shutil.ignore_patterns("yippy_cache"))
        coro = Coronagraph(root, interpolation="sinc")
        view = EqxCoronagraph(yippy_coro=coro)
        np.testing.assert_allclose(
            view.create_psf(4.0, 5.5), coro.create_psf(4.0, 5.5), rtol=1e-14
        )
        np.testing.assert_allclose(
            view.stellar_intens(0.0), coro.stellar_intens(0.0 * lod), rtol=1e-14
        )
        np.testing.assert_allclose(
            view.background_transmission(
                500.0,
                pixel_scale_lod=view.sky_pixel_scale_lod,
                shape=view.sky_trans.shape,
            ),
            coro.sky_trans(),
            atol=1e-14,
        )
        offaxis_frame = (
            float(coro.offaxis_header.lambda0.to_value("nm")),
            float(coro.offaxis_header.diameter.to_value("m")),
        )
        assert view.stellar_frame != offaxis_frame
        if importlib.util.find_spec("optixstuff") is not None:
            from optixstuff import YippyCoronagraph

            consumer = YippyCoronagraph(backend=view)
            np.testing.assert_allclose(
                consumer.stellar_map(
                    500.0,
                    0.0,
                    pixel_scale_lod=view.stellar_pixel_scale_lod,
                    shape=view.stellar_intens(0.0).shape,
                ),
                view.stellar_intens(0.0),
                atol=1e-14,
            )
            x, y = jnp.array([4.0]), jnp.array([5.5])
            np.testing.assert_allclose(
                consumer.source_psfs(
                    500.0,
                    x,
                    y,
                    pixel_scale_lod=view.pixel_scale_lod,
                    shape=view.psf_shape,
                ),
                coro.create_psfs(x, y),
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
                np.max(np.abs(np.asarray(coro.create_psf(*q)) - image)) <= 1e-5 * peak
            )


@pytest.mark.parametrize("extra_missing", [False, True])
def test_bounded_legacy_refuses(tmp_path, monkeypatch, extra_missing):
    """A bounded request cannot construct an unbounded legacy fallback."""
    sc = pytest.importorskip("sinchole")
    legacy = write_synthetic_yip(
        tmp_path / "legacy", npix=4, max_offset_lod=1.0, offset_step_lod=0.5
    )
    original = importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name == "sinchole":
            raise ModuleNotFoundError("no sinchole", name="sinchole")
        return original(name, *args, **kwargs)

    if extra_missing:
        monkeypatch.setattr(importlib, "import_module", unavailable)
    with (
        jax.enable_x64(),
        pytest.raises((ValueError, SincExtraMissing), match=r"[Bb]ounded|yippy"),
    ):
        Coronagraph(
            legacy.path,
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


@pytest.mark.parametrize("gate", ["precision", "memory", "extra"])
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
def test_validation_precedes_fallback(tmp_path, monkeypatch, gate, defect):
    """Resource routing cannot hide corrupt declarations or independent files."""
    sc = pytest.importorskip("sinchole")
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
    original = importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name == "sinchole":
            raise ModuleNotFoundError("no sinchole", name="sinchole")
        return original(name, *args, **kwargs)

    if gate == "extra":
        monkeypatch.setattr(importlib, "import_module", unavailable)
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
    sc = pytest.importorskip("sinchole")
    m = write_delivery(tmp_path, full=True)
    del (m["planes"][0] if field == "sampling_min_wavelength_m" else m)[field]
    save_manifest(tmp_path, m)
    with jax.enable_x64():
        _, backend, info = route_interpolation(tmp_path, "auto", None, None)
        assert info.reasons == ("METADATA_MISSING",)
        assert backend.create_psf_with_info(0.0, 0.0).accepted
        with pytest.raises(sc.AdmissionRefused) as caught:
            route_interpolation(tmp_path, "sinc", None, None)
        assert caught.value.reason_bits == int(sc.Reason.METADATA_MISSING)


def test_campaign_claim_refuses(tmp_path):
    """A failed explicit campaign claim needs a revised manifest to proceed."""
    pytest.importorskip("sinchole")
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
    sc = pytest.importorskip("sinchole")
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
                Coronagraph(tmp_path, interpolation="auto", sinc_options=o)
                for o in requests
            ]
            assert all(c.interpolation_info.effective == "blend" for c in objects)
            assert len({c._datacube_cache_path for c in objects}) == len(objects)
            assert len({c._perf_filename() for c in objects}) == len(objects)
            for i, coro in enumerate(objects):
                entry, reload = (
                    (coro.create_psf_datacube, coro._create_psf_datacube_gpu)
                    if i % 2 == 0
                    else (coro._create_psf_datacube_gpu, coro.create_psf_datacube)
                )
                assert not coro._datacube_cache_path.exists()
                entry(batch_size=4)
                before = np.asarray(coro.psf_datacube)
                assert np.isfinite(before).all()
                assert (coro._perf_dir / coro._perf_filename()).exists()

                def no_reads(*args, **kwargs):
                    raise AssertionError("cache reload performed source reads")

                with monkeypatch.context() as patch:
                    patch.setattr(coro, "create_psfs_with_info", no_reads)
                    reload(batch_size=4)
                np.testing.assert_array_equal(coro.psf_datacube, before)
                # Reconstructing the same request also reuses its performance entry.
                perf_path = coro._perf_dir / coro._perf_filename()
                stamp = perf_path.stat().st_mtime_ns
                again = Coronagraph(
                    tmp_path, interpolation="auto", sinc_options=requests[i]
                )
                assert again._perf_filename() == coro._perf_filename()
                assert again._datacube_cache_path == coro._datacube_cache_path
                assert perf_path.stat().st_mtime_ns == stamp


@pytest.mark.parametrize("measure,factor", [("integral", 2.0), ("mean", 0.125)])
def test_intake_finite_error_units(tmp_path, measure, factor):
    """Real intake scales nonzero finite empirical residuals with pixel values."""
    pytest.importorskip("sinchole")
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
    from scipy.special import j1

    from yippy.performance import circular_aperture

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
    sc = pytest.importorskip("sinchole")
    changed = copy.copy(admitted)
    changed.offax = eqx.tree_at(
        lambda b: b.reader.policy,
        admitted.offax,
        sc.ReadPolicy("bounded", jnp.asarray(budget)),
    )
    assert changed._datacube_cache_path != admitted._datacube_cache_path
    assert changed._perf_filename() != admitted._perf_filename()
    changed._load_or_compute_performance()
    changed._create_psf_datacube_gpu(batch_size=4)
    before = np.asarray(changed.psf_datacube)
    changed.create_psf_datacube(batch_size=4)
    np.testing.assert_array_equal(changed.psf_datacube, before)
    assert not changed.psf_datacube_info["accepted"].any()
    assert changed._datacube_cache_path.exists()
    assert (changed._perf_dir / changed._perf_filename()).exists()


def test_original_offset_precision(tmp_path):
    """The core receipt retains the delivered float32 offset precision."""
    pytest.importorskip("sinchole")
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
    "gate", ["normal", "blend", "precision", "memory", "extra", "metadata", "sampling"]
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
def test_admission_declaration_errors(tmp_path, monkeypatch, gate, defect):
    """Fails if a fallback masks hard invalid evidence or a physical conflict."""
    sc = pytest.importorskip("sinchole")
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
    original = importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name == "sinchole":
            raise ModuleNotFoundError("no sinchole", name="sinchole")
        return original(name, *args, **kwargs)

    if gate == "extra":
        monkeypatch.setattr(importlib, "import_module", unavailable)
    options = (
        SincOptions(memory=sc.MemoryBudget(1, 1, 1, 1)) if gate == "memory" else None
    )
    with jax.enable_x64(gate != "precision"), pytest.raises(ValueError):
        route_interpolation(
            tmp_path, "blend" if gate == "blend" else "auto", options, None
        )


@pytest.mark.medium
@pytest.mark.parametrize("peak", ["absent", None, 0.001])
def test_peak_reference_controls(tmp_path, monkeypatch, peak):
    """Fails if unknown or positive peak evidence prevents admission or fallback.

    Cold independent-check admission and finite reads cover all seven routes.
    """
    sc = pytest.importorskip("sinchole")
    m = write_delivery(tmp_path, full=True)
    if peak == "absent":
        del m["checks"][0]["local_peak_reference"]
    else:
        m["checks"][0]["local_peak_reference"] = peak
    original = importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name == "sinchole":
            raise ModuleNotFoundError("no sinchole", name="sinchole")
        return original(name, *args, **kwargs)

    for gate in (
        "normal",
        "blend",
        "precision",
        "memory",
        "extra",
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
        with monkeypatch.context() as patch, jax.enable_x64(gate != "precision"):
            if gate == "extra":
                patch.setattr(importlib, "import_module", unavailable)
            _, backend, info = route_interpolation(
                tmp_path, "blend" if gate == "blend" else "auto", options, None
            )
            assert info.effective == ("sinc" if gate == "normal" else "blend")
            result = backend.create_psf_with_info(0.0, 0.0)
            assert result.accepted and np.isfinite(result.intensity).all()
