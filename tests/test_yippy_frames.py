"""Independent stellar and sky frames are respected by the table contract."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from footprint_truth import assert_stellar_read, axis_positions, floored_spline_read
from optixstuff.grids import Image, PixelGrid, resample
from synthetic_yip import synthetic_coronagraph

from yippy import open_yip


def test_separate_stellar_and_sky_frames(cached_diameter_coro):
    """Fails if a table image is read on another product's grid or frame.

    The sky map is checked by exact algebra against an explicit resample of
    the sky product on its own grid. The stellar truth is the SciPy footprint
    read of the native samples on the stellar grid (``footprint_truth``), with
    the geometry typed from the grids and compared to rounding on every pixel
    (the edge as documented, see ``test_shared_frame_bit_equality``); the
    stellar grid, the sky grid and the native grid all differ, so a read on
    the wrong one moves the pixels and the scale.
    """
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
    assert stellar.grid == sg
    actual = coro.stellar_map(500.0, 0.1, grid=target).data
    assert_stellar_read(actual, stellar.data, sg, target)
    # The truth separates the grids: read on the native grid it differs.
    wrong = floored_spline_read(stellar.data, axis_positions(backend.grid, target))
    assert not np.allclose(actual, wrong, atol=0.0, rtol=1e-3)
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
    assert_stellar_read(
        coro.stellar_map(500.0, 0.1, grid=other_grid).data,
        stellar.data,
        sg.to_arcsec(500.0),
        other_grid.to_arcsec(600.0),
    )
    # Frame conversion is explicit angular geometry; no inferred area diameter.
    angular = target.to_arcsec(500.0)
    assert_stellar_read(
        coro.on_axis_psf(500.0, angular).data,
        backend.stellar_map(None, 0.0).integrated().data,
        sg.to_arcsec(500.0),
        angular,
    )


def test_shared_frame_bit_equality(cached_diameter_coro):
    """Fails if a product on its native grid changes frame, center or read.

    The sky and source products reproduce an explicit resample of the native
    Image onto its own grid bit for bit. The stellar map is a point density
    read as the footprint integral of its reconstruction even on its own grid,
    so it is not the stored sample: its truth is the SciPy footprint read of
    the native samples (``footprint_truth``), compared to rounding on every
    pixel. The fixture's stellar table is uniform, so a center or
    reconstruction change shows only in the two edge rings, where the truth
    follows the documented edge (the spline restricted to the table
    footprint, zero beyond it for the linear floor); the full array is
    compared for that reason.
    """
    c = cached_diameter_coro
    g = c.grid
    assert c.stellar_grid == g
    assert_stellar_read(
        c.stellar_map(500.0, 0.1, grid=g).data,
        c.stellar_map(None, 0.1).integrated().data,
        g,
        g,
    )
    for actual, expected in [
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
def test_simple_pupil_from_yip(unit, tmp_path):
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
        primary = open_yip(path).prepare().pupil
        assert primary.diameter("circumscribed") == 6.0
        assert primary.diameter("inscribed") == 4.7
        np.testing.assert_allclose(
            primary.collecting_area_m2,
            np.pi * 9 * (1 - obscured),
            rtol=16 * np.finfo(float).eps,
        )


@pytest.mark.parametrize("key", ["D", "D_INSC", "OBSCURED"])
def test_simple_pupil_from_yip_requires_geometry(key, tmp_path):
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
        _ = open_yip(path).prepare().pupil
