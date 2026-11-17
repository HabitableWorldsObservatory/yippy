"""Opt-in spatial readers reach public APIs without mixing cached profiles."""

from itertools import pairwise

import equinox as eqx
import jax
import numpy as np
import pytest
from astropy.io import fits
from optixstuff.apertures import Circular
from optixstuff.performance import performance_curves
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates, spline_filter
from scipy.special import j1
from synthetic_yip import write_synthetic_yip

from yippy import Coronagraph
from yippy.performance import stellar_intensity_curves


def quintic_ring_reference(image, center, radii_pix):
    """Independent scipy cardinal spline sampled on in-image circle arcs."""
    image = np.asarray(image, dtype=np.float64)
    padding = 48
    coefficients = spline_filter(
        np.pad(image, padding, mode="edge"), order=5, mode="nearest"
    )
    ny, nx = image.shape
    n_theta = 2 * max(image.shape)
    profiles = []
    for radius in np.asarray(radii_pix):
        angles = [0.0, 2 * np.pi]
        if radius > 0:
            for boundary in (0, ny - 1):
                value = np.clip((boundary - center[0]) / radius, -1, 1)
                a = np.arcsin(value)
                angles.extend([a % (2 * np.pi), (np.pi - a) % (2 * np.pi)])
            for boundary in (0, nx - 1):
                value = np.clip((boundary - center[1]) / radius, -1, 1)
                a = np.arccos(value)
                angles.extend([a, (-a) % (2 * np.pi)])
        angles = np.sort(angles)
        arcs = []
        for lo, hi in pairwise(angles):
            mid = (lo + hi) / 2
            row, col = (
                center[0] + radius * np.sin(mid),
                center[1] + radius * np.cos(mid),
            )
            if 0 <= row <= ny - 1 and 0 <= col <= nx - 1 and hi > lo:
                arcs.append((lo, hi))
        length = sum(hi - lo for lo, hi in arcs)
        if length > 0:
            positions = length * (np.arange(n_theta) + 0.5) / n_theta
            cumulative = 0.0
            theta = np.empty(n_theta)
            for lo, hi in arcs:
                selected = (positions >= cumulative) & (
                    positions < cumulative + hi - lo
                )
                theta[selected] = lo + positions[selected] - cumulative
                cumulative += hi - lo
        else:
            theta = angles
        rows = center[0] + radius * np.sin(theta)
        cols = center[1] + radius * np.cos(theta)
        inside = (
            (rows >= -1e-13)
            & (rows <= ny - 1 + 1e-13)
            & (cols >= -1e-13)
            & (cols <= nx - 1 + 1e-13)
        )
        values = map_coordinates(
            coefficients,
            [rows[inside] + padding, cols[inside] + padding],
            order=5,
            mode="nearest",
            prefilter=False,
        )
        profiles.append(np.mean(values) if values.size else 0.0)
    return np.asarray(profiles)


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
    return coro


@pytest.mark.parametrize("diameter", [0.0, 0.03])
def test_public_intensity_selects_reader(reader_coro, diameter):
    """Point and extended sources use the requested reader in both public views."""
    coro = reader_coro
    separation = np.array([1.0, 2.0, 3.0])
    with jax.enable_x64():
        keys_before = (
            stellar_intensity_curves(
                coro, getattr(diameter, "value", diameter), interpolation="keys"
            )
            .evaluate("core_mean_intensity", separation)
            .copy()
        )
        expected = quintic_ring_reference(
            coro.stellar_map(None, diameter, grid=None).data,
            (coro.stellar_grid.center_y, coro.stellar_grid.center_x),
            separation / coro.stellar_grid.step,
        )
        actual = stellar_intensity_curves(
            coro, getattr(diameter, "value", diameter), interpolation="quintic"
        ).evaluate("core_mean_intensity", separation)
        view = coro
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
            coro, getattr(diameter, "value", diameter), interpolation="keys"
        ).evaluate("core_mean_intensity", separation)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_allclose(traced, expected, rtol=1e-12)
    np.testing.assert_array_equal(
        np.asarray(keys_before).view(np.uint64), np.asarray(keys_after).view(np.uint64)
    )
    assert np.max(np.abs(keys_before / expected - 1)) > 1e-3


def test_curve_and_noise_callers_forward_reader(reader_coro):
    """Every diameter record matches independent scipy rings and PCHIP at 1e-12."""
    coro = reader_coro
    grid = coro.stellar_grid
    diameters = np.r_[0.0, np.asarray(coro._stellar_ln_interp.x)[1:]]
    with jax.enable_x64():
        for diameter in diameters:
            record = stellar_intensity_curves(coro, diameter, interpolation="quintic")
            nodes = np.asarray(record.separation_lod)
            expected = quintic_ring_reference(
                coro.stellar_map(None, diameter).data,
                (grid.center_y, grid.center_x),
                nodes / grid.step,
            )
            queries = np.sort(np.r_[nodes, (nodes[1:] + nodes[:-1]) / 2])
            actual = record.evaluate("core_mean_intensity", queries)
            np.testing.assert_allclose(
                actual, PchipInterpolator(nodes, expected)(queries), rtol=1e-12
            )


def test_rebuilt_tables_do_not_change_default_reader(reader_coro):
    """Opt-in table construction keeps later Keys requests and JAX views distinct."""
    coro = reader_coro
    separation = np.array([1.0, 2.0, 3.0])
    with jax.enable_x64():
        diameter = float(coro._stellar_ln_interp.x[-1])
        before = (
            stellar_intensity_curves(
                coro, getattr(diameter, "value", diameter), interpolation="keys"
            )
            .evaluate("core_mean_intensity", separation)
            .copy()
        )
        # Opt-in spatial profiles and aperture records have separate meanings.
        stellar_intensity_curves(coro, diameter, interpolation="quintic")
        performance_curves(
            lambda s: coro.off_axis_psf(None, s, grid=None),
            coro.stellar_map(None, diameter, grid=None),
            Circular(0.7),
            separations_lod=separation,
        )
        view = coro
        actual = np.asarray(
            eqx.filter_jit(
                (
                    lambda separation, diameter=0.0, interpolation="keys": (
                        stellar_intensity_curves(
                            view, diameter, interpolation=interpolation
                        ).evaluate("core_mean_intensity", separation)
                    )
                )
            )(separation, diameter)
        )
        np.testing.assert_allclose(actual, before, rtol=1e-12)
        np.testing.assert_allclose(
            stellar_intensity_curves(
                coro, getattr(diameter, "value", diameter), interpolation="keys"
            ).evaluate("core_mean_intensity", separation),
            before,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            stellar_intensity_curves(coro, 0.0, interpolation="quintic").evaluate(
                "core_mean_intensity", separation
            ),
            quintic_ring_reference(
                coro.stellar_map(None, 0.0).data,
                (coro.stellar_grid.center_y, coro.stellar_grid.center_x),
                separation / coro.stellar_grid.step,
            ),
            rtol=1e-12,
        )
