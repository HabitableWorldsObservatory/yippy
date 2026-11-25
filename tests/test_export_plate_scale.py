"""Check EXOSIMS plate scales and the absolute stellar leakage density."""

import json
from dataclasses import replace

import equinox as eqx
import numpy as np
import pytest
from astropy.io import fits
from hwoutils.conversions import lambda_d_to_arcsec
from sinc_fixture import save_manifest, write_delivery
from synthetic_yip import write_synthetic_yip

from yippy import open_yip
from yippy.export import export_exosims

try:
    import ineedvalidation as vv
except ModuleNotFoundError:
    vv = None


@pytest.fixture(scope="module")
def unequal_product_grids(tmp_path_factory):
    """Prepare a uniform stellar map with twice the off-axis pitch."""
    directory = tmp_path_factory.mktemp("export_plate_scale")
    metadata = write_delivery(directory)
    metadata["checks"] = []
    metadata["served"]["bounds_lod"] = [[-0.2, 0.2], [-0.2, 0.5]]
    save_manifest(directory, metadata)
    stellar = directory / "legacy" / "stellar_intens.fits"
    with fits.open(stellar, mode="update") as hdus:
        hdus[0].header["PIXSCALE"] = 0.5
        for key in ("LAMBDA", "MINLAM", "MAXLAM"):
            hdus[0].header[key] = (500.0, "nm")
    yield open_yip(directory).prepare(interpolation="sinc")


@pytest.mark.parametrize("units", ["LAMBDA/D", "arcsec"])
def test_export_preserves_stellar_density(unequal_product_grids, tmp_path, units):
    """The constant native stellar density is the absolute truth in either unit."""
    coro = unequal_product_grids
    write_to = tmp_path / "exosims"
    export_exosims(coro, units, output_dir=write_to)
    data, header = fits.getdata(write_to / "core_mean_intensity.fits", header=True)
    density_lod2 = coro.stellar_map(None, 0.0).density("grid").data[0, 0]
    angle_scale = lambda_d_to_arcsec(1, 500, 6) if units == "arcsec" else 1
    np.testing.assert_allclose(
        data[:, 1] / header["PIXSCALE"] ** 2,
        density_lod2 / angle_scale**2,
        rtol=1e-6,
        atol=0,
    )


if vv is not None:
    test_export_preserves_stellar_density = vv.seam(
        "yippy.Coronagraph", "yippy.export_exosims"
    )(test_export_preserves_stellar_density)


def test_arcsec_export_plate_scale_by_product(unequal_product_grids, tmp_path):
    """Verify angular pitch against hand conversion using the FITS product grids.

    EXOSIMS/Prototypes/OpticalSystem.py lines 119-122 document core_platescale.
    Lines 695-701 select core_platescale and core_platescale_units for each system.
    Lines 751-765 interpret LAMBDA/D as wavelength over pupil diameter and parse
    arcsec as an angular unit, converting the plate scale to arcsec internally.
    Lines 1163-1189 give the core intensity FITS header precedence, and line 2103
    divides by the plate scale squared to recover the stellar leakage density.
    """
    coro = unequal_product_grids
    write_to = tmp_path / "exosims"
    export_exosims(coro, "arcsec", output_dir=write_to)
    arcsec_per_lod = lambda_d_to_arcsec(1, 500, 6)
    for name, grid in (
        ("core_area", coro.grid),
        ("core_thruput", coro.grid),
        ("raw_contrast", coro.grid),
        ("core_mean_intensity", coro.stellar_grid),
        ("occ_trans", coro.sky_trans.grid),
    ):
        header = fits.getheader(write_to / f"{name}.fits")
        assert header["UNITS"] == "arcsec"
        assert "arcsec" in header.comments["PIXSCALE"]
        pitch_arcsec = lambda_d_to_arcsec(
            grid.step, grid.frame.reference_wavelength_nm, grid.frame.diameter_m
        )
        np.testing.assert_allclose(header["PIXSCALE"], pitch_arcsec, rtol=1e-6, atol=0)

    system = json.loads((write_to / "specs.json").read_text())[
        "starlightSuppressionSystems"
    ][0]
    assert system["core_platescale_units"] == "arcsec"
    np.testing.assert_allclose(
        system["core_platescale"],
        coro.stellar_grid.step * arcsec_per_lod,
        rtol=1e-6,
        atol=0,
    )


def test_export_refuses_dark_point_star_before_writing(tmp_path):
    """A dark plane stays invalid even when other stellar diameters are bright."""
    package_dir = tmp_path / "package"
    write_synthetic_yip(package_dir, npix=24, max_offset_lod=2.0)
    with fits.open(package_dir / "stellar_intens.fits", mode="update") as hdus:
        hdus[0].data[0] = 0
    coro = open_yip(package_dir).prepare()

    write_to = tmp_path / "exosims"
    write_to.mkdir()
    with pytest.raises(ValueError, match=r"leakage.*separations.*0\.5.*0\.75.*1\.0"):
        export_exosims(coro, separations_lod=[0.5, 0.75, 1.0], output_dir=write_to)
    assert list(write_to.iterdir()) == []


def test_export_names_separation_with_nonfinite_leakage(
    unequal_product_grids, tmp_path
):
    """A rising leakage profile overflows when extrapolated far past the knots."""
    coro = unequal_product_grids
    record = coro.curves()
    axis = np.asarray(record.separation_lod)
    levels = np.geomspace(1e-50, 1, axis.size)
    leakage = np.broadcast_to(levels, record.leakage_density_lod2.shape)
    record = replace(record, leakage_density_lod2=leakage)
    coro = eqx.tree_at(lambda model: model._records[0], coro, record)

    write_to = tmp_path / "exosims"
    with pytest.raises(ValueError, match=r"leakage.*separations.*\[10000\.0\]"):
        export_exosims(coro, separations_lod=[axis[-1], 10000.0], output_dir=write_to)
    assert not write_to.exists()
