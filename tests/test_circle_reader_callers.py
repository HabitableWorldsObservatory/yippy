"""Opt-in spatial readers reach public APIs without mixing cached profiles."""

import equinox as eqx
import jax
import numpy as np
import pytest
from astropy.io import fits
from lod_unit import lod
from scipy.special import j1
from synthetic_yip import write_synthetic_yip

from yippy import Coronagraph, EqxCoronagraph
from yippy.performance import azimuthal_mean_profile, stellar_axis_index


@pytest.fixture(scope="module", params=[(None, False), (1, False), (1, True)])
def reader_coro(request, tmp_path_factory):
    """A small real package with nonconstant stellar maps and both radial readers."""
    package = write_synthetic_yip(
        tmp_path_factory.mktemp("circle-callers"),
        npix=32,
        pixscale=0.25,
        max_offset_lod=3.5,
    )
    order, single_diameter = request.param
    with fits.open(package.path / "stellar_intens.fits", mode="update") as hdul:
        y, x = np.indices((32, 32)) - 15.5
        z = np.pi * np.hypot(y, x) * 0.25
        image = (2 * j1(z) / z) ** 2 + 1e-8
        levels = (
            1
            if single_diameter
            else (1 + np.arange(hdul[0].data.shape[0]))[:, None, None]
        )
        hdul[0].data[:] = image * levels
    with jax.enable_x64():
        coro = Coronagraph(package.path, interp_order=order)
    if single_diameter:
        # Exercise the one-profile fast path with valid spline support. The
        # existing FITS loader requires >=2 diameter knots even for a constant
        # map; advertising one available diameter does not change that map.
        coro.stellar_intens.diams = coro.stellar_intens.diams[:1]
    return coro


@pytest.mark.parametrize("diameter", [0.0, 0.03])
def test_public_intensity_selects_reader(reader_coro, diameter):
    """Point and extended sources use the requested reader in both public views."""
    coro = reader_coro
    separation = np.array([1.0, 2.0, 3.0])
    with jax.enable_x64():
        keys_before = coro.core_mean_intensity(separation, diameter).copy()
        expected = np.asarray(
            eqx.filter_jit(azimuthal_mean_profile)(
                coro.stellar_intens(diameter * lod),
                stellar_axis_index(coro),
                separation / coro.pixel_scale_arcsec.value,
                interpolation="quintic",
            )
        )
        actual = coro.core_mean_intensity(separation, diameter, interpolation="quintic")
        view = EqxCoronagraph(yippy_coro=coro)
        traced = np.asarray(
            eqx.filter_jit(view.core_mean_intensity)(
                separation, diameter, interpolation="quintic"
            )
        )
        keys_after = coro.core_mean_intensity(separation, diameter)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_allclose(traced, expected, rtol=1e-12)
    np.testing.assert_array_equal(
        keys_before.view(np.uint64), keys_after.view(np.uint64)
    )
    assert np.max(np.abs(keys_before / expected - 1)) > 1e-3


def test_curve_and_noise_callers_forward_reader(reader_coro):
    """Curve, map and noise helpers propagate the opt-in spatial reader."""
    coro = reader_coro
    with jax.enable_x64():
        sep, curves = coro.core_mean_intensity_curve(
            plot=False, interpolation="quintic"
        )
        diameter = coro.stellar_intens.diams[0]
        np.testing.assert_allclose(
            coro.core_mean_intensity(sep, diameter, interpolation="quintic"),
            curves[diameter],
            rtol=1e-12,
        )
        expected = coro.core_mean_intensity(
            np.array([1.0, 2.0, 3.0]), interpolation="quintic"
        )
        np.testing.assert_allclose(
            coro.noise_floor_ayo(np.array([1.0, 2.0, 3.0]), interpolation="quintic")
            * 30,
            expected,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            coro.noise_floor_ayo_map(interpolation="quintic") * 30,
            coro.core_mean_intensity_map(interpolation="quintic"),
            rtol=1e-12,
        )


def test_rebuilt_tables_do_not_change_default_reader(reader_coro):
    """Opt-in table construction keeps later Keys requests and JAX views distinct."""
    coro = reader_coro
    separation = np.array([1.0, 2.0, 3.0])
    with jax.enable_x64():
        diameter = coro.stellar_intens.diams[-1]
        before = coro.core_mean_intensity(separation, diameter).copy()
        coro.compute_all_performance_curves(
            oversample=1,
            save_to_fits=False,
            interp_order=coro.interp_order,
            interpolation="quintic",
            stellar_diam=diameter,
        )
        view = EqxCoronagraph(yippy_coro=coro)
        actual = np.asarray(eqx.filter_jit(view.core_mean_intensity)(separation))
        np.testing.assert_allclose(actual, before, rtol=1e-12)
        np.testing.assert_allclose(
            coro.core_mean_intensity(separation), before, rtol=1e-12
        )
        np.testing.assert_allclose(
            coro.core_mean_intensity(separation, interpolation="quintic"),
            view.core_mean_intensity(separation, interpolation="quintic"),
            rtol=1e-12,
        )
