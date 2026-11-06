"""Optional off-axis routing, geometry and refusal regression tests."""

import copy
import importlib
import json
import os
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
        save_manifest(root, metadata)

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
    m["planes"][0]["psf_path"] = "not-selected.fits"
    with fits.open(tmp_path / "nodes.fits", mode="update") as hdus:
        hdus[0].header["LAMBDA"] = (0.5, "um")
        hdus[0].header["D"] = (600.0, "cm")
        hdus[0].header["PIXSCALE"] = (m["pixel_scale_arcsec_xy"][0], "arcsec/pixel")
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
    with pytest.raises((sc.AdmissionRefused, ValueError)):
        route_interpolation(tmp_path, "auto", None, None)
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
    x, y = jnp.array([0.07]), jnp.array([-0.04])
    expected = admitted.create_psfs(x, y)
    result = consumer.source_psfs(500.0, x, y, pixel_scale_lod=0.25, shape=(4, 4))
    np.testing.assert_allclose(result, expected, atol=1e-18)
    np.testing.assert_array_equal(
        view.source_psfs(600.0, x, y, pixel_scale_lod=0.25, shape=(4, 4)), result
    )


def test_cache_identity_and_selections(admitted):
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
    admitted.create_psf_datacube(batch_size=4)
    before = np.asarray(admitted.psf_datacube)
    selected = admitted.psf_datacube_info["selected_backend"].copy()
    admitted._create_psf_datacube_gpu(batch_size=4)
    np.testing.assert_array_equal(admitted.psf_datacube, before)
    np.testing.assert_array_equal(
        admitted.psf_datacube_info["selected_backend"], selected
    )
    assert not admitted.psf_datacube_info["accepted"].all()
    base_cube.with_suffix(".status.npz").unlink()
    admitted.create_psf_datacube(batch_size=4)
    assert base_cube.with_suffix(".status.npz").exists()


@pytest.mark.slow
def test_public_delivery():
    """Fails if independent images disagree by over 1e-5 of their local peaks."""
    pytest.importorskip("sinchole")
    configured = os.environ.get("YIPPY_SINC_DELIVERY")
    if not configured or not (Path(configured) / "sinc-delivery.json").exists():
        pytest.skip("Set YIPPY_SINC_DELIVERY to an independent intensity delivery")
    with jax.enable_x64():
        root = Path(configured)
        coro = Coronagraph(root, interpolation="sinc")
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
