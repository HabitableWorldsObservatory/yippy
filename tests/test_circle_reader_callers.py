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
from yippy.performance import (
    azimuthal_mean_profile,
    compute_core_mean_intensity_curve,
    stellar_intensity_curves,
)


@pytest.fixture(
    scope="module",
    params=[
        ("aperture_mean", False),
        ("azimuthal_ring_mean", False),
        ("aperture_mean", True),
    ],
)
def reader_coro(request, tmp_path_factory):
    """A small real package with nonconstant stellar maps and both radial readers."""
    package = write_synthetic_yip(
        tmp_path_factory.mktemp("circle-callers"),
        npix=32,
        pixscale=0.25,
        max_offset_lod=3.5,
    )
    leakage_rule, single_diameter = request.param
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
        coro = Coronagraph(package.path, leakage_rule=leakage_rule)
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
        keys_before = (
            stellar_intensity_curves(
                EqxCoronagraph(yippy_coro=coro),
                getattr(diameter, "value", diameter),
                interpolation="keys",
            )
            .evaluate("core_mean_intensity", separation)
            .copy()
        )
        expected = np.asarray(
            eqx.filter_jit(azimuthal_mean_profile)(
                coro.stellar_intens(diameter * lod).data,
                (coro.stellar_intens.grid.center_y, coro.stellar_intens.grid.center_x),
                separation / coro.pixel_scale_arcsec.value,
                interpolation="quintic",
            )
        )
        actual = stellar_intensity_curves(
            EqxCoronagraph(yippy_coro=coro),
            getattr(diameter, "value", diameter),
            interpolation="quintic",
        ).evaluate("core_mean_intensity", separation)
        view = EqxCoronagraph(yippy_coro=coro)
        traced = np.asarray(
            eqx.filter_jit(
                (
                    lambda separation, diameter=0.0, interpolation="keys": (
                        stellar_intensity_curves(
                            view, diameter, interpolation=interpolation
                        ).evaluate("core_mean_intensity", separation)
                    )
                )
            )(separation, diameter, interpolation="quintic")
        )
        keys_after = stellar_intensity_curves(
            EqxCoronagraph(yippy_coro=coro),
            getattr(diameter, "value", diameter),
            interpolation="keys",
        ).evaluate("core_mean_intensity", separation)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_allclose(traced, expected, rtol=1e-12)
    np.testing.assert_array_equal(
        np.asarray(keys_before).view(np.uint64), np.asarray(keys_after).view(np.uint64)
    )
    assert np.max(np.abs(keys_before / expected - 1)) > 1e-3


def test_curve_and_noise_callers_forward_reader(reader_coro):
    """Curve, map and noise helpers propagate the opt-in spatial reader."""
    coro = reader_coro
    with jax.enable_x64():
        sep, curves = compute_core_mean_intensity_curve(coro, interpolation="quintic")
        diameter = coro.stellar_intens.diams[0]
        np.testing.assert_allclose(
            stellar_intensity_curves(
                EqxCoronagraph(yippy_coro=coro),
                getattr(diameter, "value", diameter),
                interpolation="quintic",
            ).evaluate("core_mean_intensity", sep),
            curves[diameter],
            rtol=1e-12,
        )
        expected = stellar_intensity_curves(
            EqxCoronagraph(yippy_coro=coro), 0.0, interpolation="quintic"
        ).evaluate("core_mean_intensity", np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(
            (
                stellar_intensity_curves(
                    EqxCoronagraph(yippy_coro=coro), 0.0, interpolation="quintic"
                ).evaluate("core_mean_intensity", np.array([1.0, 2.0, 3.0]))
                / 30
            )
            * 30,
            expected,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            (
                stellar_intensity_curves(
                    EqxCoronagraph(yippy_coro=coro), 0.0, interpolation="quintic"
                ).evaluate(
                    "core_mean_intensity",
                    np.hypot(*coro.stellar_intens.grid.coordinates()),
                )
                / 30.0
            )
            * 30,
            stellar_intensity_curves(
                EqxCoronagraph(yippy_coro=coro), 0.0, interpolation="quintic"
            ).evaluate(
                "core_mean_intensity", np.hypot(*coro.stellar_intens.grid.coordinates())
            ),
            rtol=1e-12,
        )


def test_rebuilt_tables_do_not_change_default_reader(reader_coro):
    """Opt-in table construction keeps later Keys requests and JAX views distinct."""
    coro = reader_coro
    separation = np.array([1.0, 2.0, 3.0])
    with jax.enable_x64():
        diameter = coro.stellar_intens.diams[-1]
        before = (
            stellar_intensity_curves(
                EqxCoronagraph(yippy_coro=coro),
                getattr(diameter, "value", diameter),
                interpolation="keys",
            )
            .evaluate("core_mean_intensity", separation)
            .copy()
        )
        # Opt-in spatial profiles and aperture records have separate meanings.
        compute_core_mean_intensity_curve(coro, interpolation="quintic")
        coro.compute_all_performance_curves(
            save_to_fits=False,
            stellar_diameter=diameter,
        )
        view = EqxCoronagraph(yippy_coro=coro)
        actual = np.asarray(
            eqx.filter_jit(
                (
                    lambda separation, diameter=0.0, interpolation="keys": (
                        stellar_intensity_curves(
                            view, diameter, interpolation=interpolation
                        ).evaluate("core_mean_intensity", separation)
                    )
                )
            )(separation, diameter.to_value(lod))
        )
        np.testing.assert_allclose(actual, before, rtol=1e-12)
        np.testing.assert_allclose(
            stellar_intensity_curves(
                EqxCoronagraph(yippy_coro=coro),
                getattr(diameter, "value", diameter),
                interpolation="keys",
            ).evaluate("core_mean_intensity", separation),
            before,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            stellar_intensity_curves(
                EqxCoronagraph(yippy_coro=coro), 0.0, interpolation="quintic"
            ).evaluate("core_mean_intensity", separation),
            stellar_intensity_curves(view, 0.0, interpolation="quintic").evaluate(
                "core_mean_intensity", separation
            ),
            rtol=1e-12,
        )
