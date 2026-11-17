"""Exported performance values obey the record's single interpolation policy."""

from dataclasses import replace

import astropy.io.fits as fits
import astropy.units as u
import equinox as eqx
import numpy as np
from hwoutils.grids import Image
from optixstuff.performance import performance_curves

from yippy.export import export_ayo_csv, export_exosims


def test_exports_agree_with_record_between_knots(syn_coro_1d, tmp_path):
    """Fails if either export uses a separate interpolator or rounds CSV values.

    FITS binary doubles and 17-digit CSV text preserve evaluated float64 values;
    four epsilons allow only floating-point parsing/serialization roundoff.
    """
    coro = eqx.tree_at(lambda c: c.yip_path, syn_coro_1d, tmp_path)
    record = coro.curves()
    knots = np.asarray(record.separation_lod, dtype=np.float64)
    separations = (knots[:-1] + knots[1:]) / 2
    specs = export_exosims(coro, separations_lod=separations)
    csv_path = export_ayo_csv(coro, tmp_path / "ayo.csv", separations_lod=separations)
    csv_data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    tolerance = 4 * np.finfo(np.float64).eps

    for filename, quantity, csv_column in (
        ("raw_contrast", "raw_contrast", 1),
        ("core_thruput", "throughput", 3),
    ):
        data, header = fits.getdata(
            tmp_path / "exosims" / f"{filename}.fits", header=True
        )
        expected = record.evaluate(quantity, separations)
        np.testing.assert_array_equal(data[:, 0], separations)
        np.testing.assert_allclose(data[:, 1], expected, rtol=tolerance, atol=0)
        np.testing.assert_allclose(
            csv_data[:, csv_column], expected, rtol=tolerance, atol=0
        )
        assert header["APERRULE"] == type(record.rule).__name__
        assert header["CENTRE"] == record.rule.centre
        assert header["LEAKRULE"] == record.leakage_rule

    np.testing.assert_allclose(
        fits.getdata(tmp_path / "exosims/core_area.fits")[:, 1],
        record.evaluate("core_area_lod2", separations),
        rtol=tolerance,
        atol=0,
    )
    np.testing.assert_allclose(
        fits.getdata(tmp_path / "exosims/core_mean_intensity.fits")[:, 1],
        record.evaluate("leakage_density_lod2", separations)
        * coro.stellar_grid.pixel_area,
        rtol=tolerance,
        atol=0,
    )
    system = specs["starlightSuppressionSystems"][0]
    assert system["core_area"] == "core_area.fits"
    assert "OWA" not in system


def test_arcsec_export_converts_separation_and_area(syn_coro_1d, tmp_path):
    """Fails if angular export relabels lod tables or omits the area scale square.

    Both sides perform one angular multiplication (two for area); four float64
    epsilons cover arithmetic and FITS serialization, not a unit discrepancy.
    """
    coro = eqx.tree_at(lambda c: c.yip_path, syn_coro_1d, tmp_path)
    record = coro.curves()
    separations = np.array([1.125, 2.375, 4.125])
    specs = export_exosims(coro, "arcsec", separations_lod=separations)
    scale = float(record.frame.lod_to_arcsec(coro.header.lambda0.to_value(u.nm)))
    tolerance = 4 * np.finfo(np.float64).eps
    for filename in (
        "core_thruput",
        "core_area",
        "raw_contrast",
        "core_mean_intensity",
        "occ_trans",
    ):
        data = fits.getdata(tmp_path / "exosims" / f"{filename}.fits")
        np.testing.assert_allclose(
            data[:, 0], separations * scale, rtol=tolerance, atol=0
        )
    np.testing.assert_allclose(
        fits.getdata(tmp_path / "exosims/core_area.fits")[:, 1],
        record.evaluate("core_area_lod2", separations) * scale**2,
        rtol=tolerance,
        atol=0,
    )
    system = specs["starlightSuppressionSystems"][0]
    np.testing.assert_allclose(
        system["IWA"], record.iwa_lod * scale, rtol=tolerance, atol=0
    )
    assert "OWA" not in system

    # Double the stellar reference diameter: a planet-frame lod covers four
    # stellar-frame lod2. A constant stellar map must still export its original
    # per-pixel fraction, not four times that fraction. The 32-epsilon bound
    # covers FFT quadrature and log interpolation of a constant profile.
    stellar_grid = replace(
        coro.stellar_grid,
        frame=replace(record.frame, diameter_m=2 * record.frame.diameter_m),
    )
    coro = eqx.tree_at(lambda c: c.stellar_grid, coro, stellar_grid)
    stellar_level = 2.0**-30
    stellar = Image(
        np.full(stellar_grid.shape, stellar_level),
        stellar_grid,
        (),
        "per_pixel",
        "stellar_flux_fraction",
    )
    separations = np.array([1.125, 2.125, 2.875])
    updated = performance_curves(
        lambda s: coro.off_axis_psf(None, s, None),
        stellar,
        record.rule,
        separations_lod=separations,
    )
    coro = eqx.tree_at(lambda c: c._records, coro, (updated,))
    midpoints = (separations[:-1] + separations[1:]) / 2
    export_exosims(coro, "arcsec", separations_lod=midpoints)
    np.testing.assert_allclose(
        fits.getdata(tmp_path / "exosims/core_mean_intensity.fits")[:, 1],
        stellar_level,
        rtol=32 * np.finfo(np.float64).eps,
        atol=0,
    )


def test_log_interpolation_and_explicit_contrast_floor(syn_coro_1d, tmp_path):
    """Fails for the wrong interpolation space or an unrequested contrast floor.

    Two knots have an analytic midpoint: the arithmetic mean for throughput,
    geometric mean for contrast-like fields. Sixteen float64 epsilons cover
    logarithm/exponentiation; the wrong interpolation space differs by order one.
    """
    coro = eqx.tree_at(lambda c: c.yip_path, syn_coro_1d, tmp_path)
    updated = replace(
        coro.curves(),
        separation_lod=np.array([1.0, 3.0]),
        throughput=np.array([0.2, 0.6]),
        core_area_lod2=np.array([1.0, 1.0]),
        leakage_fraction=np.array([1e-08, 1e-10]),
        leakage_density_lod2=np.array([1e-08, 1e-10]),
        raw_contrast=np.array([1e-11, 1e-13]),
    )
    coro = eqx.tree_at(lambda c: c._records, coro, (updated,))
    record = coro.curves()
    separations = np.array([2.0])
    tolerance = 16 * np.finfo(np.float64).eps
    np.testing.assert_allclose(
        record.evaluate("throughput", separations), [0.4], rtol=tolerance
    )
    for quantity, midpoint in (
        ("raw_contrast", 1e-12),
        ("leakage_fraction", 1e-9),
        ("leakage_density_lod2", 1e-9),
    ):
        np.testing.assert_allclose(
            record.evaluate(quantity, separations), [midpoint], rtol=tolerance, atol=0
        )

    plain = export_ayo_csv(coro, tmp_path / "plain.csv", separations_lod=separations)
    floored = export_ayo_csv(
        coro, tmp_path / "floor.csv", contrast_floor=1e-10, separations_lod=separations
    )
    plain_data = np.loadtxt(plain, delimiter=",", skiprows=1, ndmin=2)
    floor_data = np.loadtxt(floored, delimiter=",", skiprows=1, ndmin=2)
    np.testing.assert_allclose(plain_data[:, 1], [1e-12], rtol=tolerance, atol=0)
    np.testing.assert_array_equal(floor_data[:, 1], [1e-10])
    np.testing.assert_allclose(floor_data[:, 2], [1e-10 / 30], rtol=tolerance, atol=0)
    np.testing.assert_array_equal(record.raw_contrast, [1e-11, 1e-13])
