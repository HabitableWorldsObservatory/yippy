"""Independent stellar and sky frames are respected by the table contract."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from optixstuff.grids import Image, PixelGrid, resample
from synthetic_yip import synthetic_coronagraph

from yippy import open_yip


def test_separate_stellar_and_sky_frames(cached_diameter_coro):
    """Exact algebra: table images and explicit resampling use each product grid."""
    backend = cached_diameter_coro
    sg = PixelGrid.regular(backend.grid.shape, 0.4, "lod", frame=backend.grid.frame)
    kg = PixelGrid.regular(backend.grid.shape, 0.3, "lod", frame=backend.grid.frame)
    sky = replace(backend.sky_trans, grid=kg)
    backend = eqx.tree_at(lambda b: (b.stellar_grid, b.sky_trans), backend, (sg, sky))
    coro = backend
    target = PixelGrid.regular(
        (20, 22), 0.35, "lod", frame=backend.grid.frame, center=(10.2, 9.7)
    )
    stellar = backend.stellar_map(None, 0.1).integrated()
    expected = resample(stellar, target)
    np.testing.assert_array_equal(
        coro.stellar_map(500.0, 0.1, grid=target).data, expected.data
    )
    wrong = resample(replace(stellar, grid=backend.grid), target)
    assert not np.allclose(expected.data, wrong.data, atol=0.0, rtol=1e-3)
    np.testing.assert_array_equal(
        coro.uniform_sky_map(500.0, grid=target).data,
        resample(sky, target).data,
    )
    # A different physical diameter and reference wavelength define another lod.
    other_frame = replace(
        backend.grid.frame, diameter_m=4.0, reference_wavelength_nm=600.0
    )
    separate_stellar = eqx.tree_at(
        lambda b: b.stellar_grid, coro, replace(sg, frame=other_frame)
    )
    assert separate_stellar.at_wavelength(500.0).stellar_grid.frame == other_frame
    assert (
        separate_stellar.at_wavelength(500.0).stellar_grid.frame
        != separate_stellar.grid.frame
    )
    other_grid = replace(target, frame=other_frame)
    np.testing.assert_array_equal(
        coro.stellar_map(500.0, 0.1, grid=other_grid).data,
        resample(
            replace(stellar, grid=sg.to_arcsec(500.0)), other_grid.to_arcsec(600.0)
        ).data,
    )
    # Frame conversion is explicit angular geometry; no inferred area diameter.
    angular = target.to_arcsec(500.0)
    np.testing.assert_allclose(
        coro.on_axis_psf(500.0, angular).data,
        resample(
            replace(
                backend.stellar_map(None, 0.0).integrated(), grid=sg.to_arcsec(500.0)
            ),
            angular,
        ).data,
        rtol=8 * np.finfo(np.float32).eps,
    )


def test_shared_frame_bit_equality(cached_diameter_coro):
    """On the native grid, products are box integrals with no centre change.

    The expected value resamples the native Image onto its own grid; a frame,
    centre or reconstruction change on the consumer side breaks bit equality.
    """
    c = cached_diameter_coro
    g = c.grid
    for actual, expected in [
        (c.stellar_map(500.0, 0.1, grid=g), c.stellar_map(None, 0.1).integrated()),
        (c.uniform_sky_map(500.0, grid=g), c.sky_trans),
        (
            c.source_psfs(500.0, jnp.array([1.0]), jnp.array([0.0]), grid=g).image,
            c.source_psfs(
                500.0, jnp.array([1.0]), jnp.array([0.0]), grid=None
            ).image.integrated(),
        ),
    ]:
        assert isinstance(actual, Image)
        np.testing.assert_array_equal(actual.data, resample(expected, g).data)


@pytest.mark.parametrize("unit", ["m", "cm"])
def test_simple_primary_from_yip(unit, tmp_path):
    """Exact header diameters; changing collecting area never changes D_INSC.

    Floating-point basis: 16 float64 eps for geometric area; header lengths
    agree after explicit metric conversion. No extra aperture factor is implicit.
    """
    path = tmp_path / "geometry"
    synthetic_coronagraph(path)
    scale = 1.0 if unit == "m" else 100.0
    for obscured in (0.2, 0.7):
        with fits.open(path / "stellar_intens.fits", mode="update") as hdus:
            hdus[0].header["D"] = (6.0 * scale, f"diameter in {unit}")
            hdus[0].header["D_INSC"] = (4.7 * scale, f"inscribed diameter in {unit}")
            hdus[0].header["OBSCURED"] = obscured
        primary = open_yip(path).prepare().primary
        assert primary.diameter("circumscribed") == 6.0
        assert primary.diameter("inscribed") == 4.7
        np.testing.assert_allclose(
            primary.collecting_area_m2,
            np.pi * 9 * (1 - obscured),
            rtol=16 * np.finfo(float).eps,
        )


@pytest.mark.parametrize("key", ["D", "D_INSC", "OBSCURED"])
def test_simple_primary_from_yip_requires_geometry(key, tmp_path):
    """Missing geometry raises instead of deriving a diameter from collecting area."""
    path = tmp_path / "geometry"
    synthetic_coronagraph(path)
    with fits.open(path / "stellar_intens.fits", mode="update") as hdus:
        del hdus[0].header[key]
    with pytest.raises(
        ValueError,
        match={
            "D": "D|diameter",
            "D_INSC": "diameter_inscribed",
            "OBSCURED": "obscured",
        }[key],
    ):
        _ = open_yip(path).prepare().primary
