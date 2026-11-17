"""What interpolation between tabulated offsets does to flux, and where the axis is.

The reader synthesizes a PSF at any position by combining the tabulated ones,
and a consumer treats the result as a photometric quantity. Two things have to
hold for that to be safe: the combination must not create or destroy light
relative to the PSFs it combined, and every path through the reader must agree
about where the optical axis sits.

The separation map, the core mean intensity profile, and off-axis synthesis
all place the axis at index (npix - 1) / 2, half a pixel before the header's
XCENTER, which counts pixels from the array's outer corner.
"""

import jax
import numpy as np
import pytest
from lod_unit import lod
from optixstuff import YippyCoronagraph

from yippy import EqxCoronagraph
from yippy.performance import stellar_intensity_curves

# Positions between tabulated knots, where the reader has to interpolate. The
# knot spacing of the synthetic package is 0.5 lambda/D.
KNOT_SPACING_LOD = 0.5
BETWEEN_KNOTS_LOD = (1.25, 2.25, 3.75, 4.25)


def _sum(coro, x_lod, y_lod=0.0):
    """Total flux of the synthesized PSF at a sky position."""
    return float(np.asarray(coro.offax(x_lod * lod, y_lod * lod).data).sum())


def test_interpolated_flux_is_bracketed_by_its_two_neighbours(syn_coro_1d):
    """A PSF between two knots holds between their two fluxes.

    The radial path combines the two bracketing PSFs with weights that sum to
    one, so the result cannot hold more light than the brighter neighbour or
    less than the dimmer one. A weighting bug that renormalized, dropped a
    neighbour, or let the weights sum to something other than one shows up
    here and nowhere else, because every shape and ratio test in the suite is
    blind to an overall scale.

    The audit found the original version of this check executing zero
    assertions, so the loop counts what it did.
    """
    coro = syn_coro_1d
    checked = 0

    for separation in BETWEEN_KNOTS_LOD:
        low = np.floor(separation / KNOT_SPACING_LOD) * KNOT_SPACING_LOD
        high = np.ceil(separation / KNOT_SPACING_LOD) * KNOT_SPACING_LOD
        assert low < separation < high, "the probe landed on a knot, not between two"

        lower_sum = _sum(coro, low)
        upper_sum = _sum(coro, high)
        interpolated = _sum(coro, separation)

        assert min(lower_sum, upper_sum) <= interpolated <= max(lower_sum, upper_sum), (
            f"the PSF at {separation} holds {interpolated:.6f}, outside the "
            f"[{min(lower_sum, upper_sum):.6f}, {max(lower_sum, upper_sum):.6f}] "
            "its two bracketing knots hold"
        )
        checked += 1

    assert checked == len(BETWEEN_KNOTS_LOD), (
        "the bracketing loop skipped separations, so it asserted less than it claims"
    )


def test_flux_grows_with_separation_as_the_occulter_opens(syn_coro_1d, synthetic_1d):
    """More light gets through further out, monotonically.

    The occulter transmission is monotone in separation by construction, and
    nothing in the reader should reverse that. This is the cheapest check that
    the separation is being used with the right sign, and unlike the direction
    anchors it is insensitive to where the blob lands.
    """
    separations = np.arange(0.5, 6.0, 0.5)
    sums = np.array([_sum(syn_coro_1d, float(s)) for s in separations])

    assert np.all(np.diff(sums) > 0), (
        f"synthesized flux is not monotone in separation: {sums}"
    )
    expected_ratio = float(synthetic_1d.occulter(5.0) / synthetic_1d.occulter(1.0))
    got_ratio = _sum(syn_coro_1d, 5.0) / _sum(syn_coro_1d, 1.0)
    assert got_ratio == pytest.approx(expected_ratio, rel=0.02), (
        "the ratio of synthesized fluxes does not follow the occulter law"
    )


def test_a_tabulated_offset_is_returned_without_interpolation(
    syn_coro_1d, synthetic_1d
):
    """At a knot the reader should hand back what the package holds.

    A query that lands exactly on a tabulated offset needs no combination, and
    the total flux there is the package's own value for that PSF. This
    separates an interpolation error from a loading error: if this fails, the
    problem is upstream of any weighting.
    """
    package = synthetic_1d
    for separation in (1.0, 2.0, 3.0):
        expected = float(package.occulter(separation) * package.psf_flux)
        got = _sum(syn_coro_1d, separation)
        # The array truncates the Gaussian's wings, so the sum is slightly
        # below the analytic total flux; well inside the field that deficit is
        # far smaller than a percent.
        assert got == pytest.approx(expected, rel=0.01), (
            f"the PSF tabulated at {separation} holds {got:.6f}, not the "
            f"{expected:.6f} the package was written with"
        )


def test_the_separation_map_is_measured_about_the_optical_axis(
    syn_coro_1d, synthetic_1d
):
    """Every pixel reads its distance from index (npix - 1) / 2 in lambda/D.

    The header's XCENTER = npix / 2 counts pixels from the array's outer
    corner, so the axis falls between the four central pixels. Those four read
    sqrt(2) / 2 pixels, not zero, and the map is point-symmetric about the
    array's middle, the same axis the core mean intensity profile uses.
    """
    with jax.enable_x64():
        npix = synthetic_1d.npix
        pixel_lod = synthetic_1d.pixscale
        axis = (npix - 1) / 2.0
        rows, columns = np.indices((npix, npix))
        expected = np.hypot(rows - axis, columns - axis) * pixel_lod

        separation_map = np.asarray(
            YippyCoronagraph(
                backend=EqxCoronagraph(yippy_coro=syn_coro_1d)
            ).separation_map()
        )

        np.testing.assert_allclose(separation_map, expected, rtol=1e-12, atol=0.0)
        middle = npix // 2
        central = separation_map[middle - 1 : middle + 1, middle - 1 : middle + 1]
        np.testing.assert_allclose(central, np.sqrt(0.5) * pixel_lod, rtol=1e-12)


def test_both_paths_agree_about_where_the_axis_is(syn_coro_1d, synthetic_1d):
    """The map and synthesis share an axis, to the 1e-5 pixel float32 fit floor."""
    separation_map = np.asarray(
        YippyCoronagraph(
            backend=EqxCoronagraph(yippy_coro=syn_coro_1d)
        ).separation_map()
    )
    nearest = np.argwhere(separation_map == separation_map.min())
    map_center = float(np.mean(nearest[:, 1]))

    columns = []
    for separation in (2.0, 3.0):
        for sign in (+1.0, -1.0):
            psf = np.asarray(syn_coro_1d.offax(sign * separation * lod, 0.0 * lod).data)
            row, col = np.unravel_index(np.argmax(psf), psf.shape)
            a, b, c = np.log(psf[row, col - 1 : col + 2])
            columns.append(col + 0.5 * (a - c) / (a - 2 * b + c))
    synthesis_center = float(np.mean(columns))

    assert synthesis_center == pytest.approx(float(map_center), abs=1e-5, rel=0), (
        f"the separation map is centered on column {map_center} while off-axis "
        f"synthesis is centered on {synthesis_center}"
    )


# The level of the four pixels that straddle the optical axis in the
# axis-concentrated stellar map, and the faint level everywhere else. Both are
# positive because the stellar-intensity reader interpolates in log space.
AXIS_BLOCK_LEVEL = 1.0
AXIS_BACKGROUND_LEVEL = 1e-4


@pytest.fixture(scope="module")
def axis_concentrated_coro(tmp_path_factory):
    """A package whose stellar leakage sits on the four pixels around the axis.

    The header follows the shipped packages: ``XCENTER = YCENTER = npix / 2``
    counts pixels from the array's outer corner, so with integer indices
    labeling pixel centers the optical axis is at index ``(npix - 1) / 2``,
    between four pixels. Every shipped stellar map is point-symmetric about
    that index and not about ``npix / 2``.
    """
    import astropy.io.fits as pyfits
    from synthetic_yip import write_synthetic_yip

    from yippy import Coronagraph

    package = write_synthetic_yip(tmp_path_factory.mktemp("axis") / "axis_concentrated")
    path = package.path / "stellar_intens.fits"
    with pyfits.open(path) as hdul:
        header = hdul[0].header.copy()
        maps = np.full_like(hdul[0].data, AXIS_BACKGROUND_LEVEL)
    axis = (package.npix - 1) // 2
    maps[:, axis : axis + 2, axis : axis + 2] = AXIS_BLOCK_LEVEL
    pyfits.PrimaryHDU(maps, header=header).writeto(path, overwrite=True)
    return Coronagraph(package.path)


# A smooth ring about the optical axis: radius and Gaussian width in pixels.
# The width keeps the map well sampled, so its azimuthal mean at any radius is
# the ring function itself, and the ring is narrow enough that a profile taken
# half a pixel off the axis on each axis is several percent off on its flanks.
AXIS_RING_RADIUS_PIX = 10.0
AXIS_RING_WIDTH_PIX = 3.0


def _axis_ring(r_pix):
    return AXIS_BACKGROUND_LEVEL + AXIS_BLOCK_LEVEL * np.exp(
        -0.5 * ((r_pix - AXIS_RING_RADIUS_PIX) / AXIS_RING_WIDTH_PIX) ** 2
    )


@pytest.fixture(scope="module")
def axis_ring_coro(tmp_path_factory):
    """A package whose stellar map is a smooth ring centered on the optical axis."""
    import astropy.io.fits as pyfits
    from synthetic_yip import write_synthetic_yip

    from yippy import Coronagraph

    package = write_synthetic_yip(tmp_path_factory.mktemp("ring") / "axis_ring")
    path = package.path / "stellar_intens.fits"
    with pyfits.open(path) as hdul:
        header = hdul[0].header.copy()
        shape = hdul[0].data.shape
    axis = (package.npix - 1) / 2
    yy, xx = np.mgrid[: shape[-2], : shape[-1]]
    ring = _axis_ring(np.hypot(yy - axis, xx - axis))
    maps = np.broadcast_to(ring, shape).copy()
    pyfits.PrimaryHDU(maps, header=header).writeto(path, overwrite=True)
    return Coronagraph(package.path)


def test_core_mean_intensity_is_measured_about_the_optical_axis(axis_ring_coro):
    """The core mean intensity is the ring's own radial function.

    The map is a ring about the optical axis, so its azimuthal mean about the
    axis at separation s is the ring function at s. A profile measured about
    a center half a pixel off on each axis smears the ring and is off by up
    to six percent on its flanks, far outside the tolerance.
    """
    coro = axis_ring_coro
    pixel_lod = float(coro.pixel_scale_arcsec.value)
    radii_pix = AXIS_RING_RADIUS_PIX + AXIS_RING_WIDTH_PIX * np.array(
        [-2.0, -1.0, -0.5, 0.0, 0.3, 0.5, 1.0, 2.0]
    )
    got = np.array(
        [
            stellar_intensity_curves(
                EqxCoronagraph(yippy_coro=coro), 0.0, interpolation="keys"
            ).evaluate("core_mean_intensity", r * pixel_lod)
            for r in radii_pix
        ]
    )
    np.testing.assert_allclose(
        got,
        _axis_ring(radii_pix),
        rtol=3e-3,
        err_msg="the core mean intensity is not the azimuthal mean about the axis",
    )


def test_core_mean_intensity_map_is_the_profile_about_the_optical_axis(
    axis_concentrated_coro,
):
    """The 2D projection of the core mean intensity is centered on the axis.

    The four pixels around the axis are equidistant from it, so they read the
    same value, the profile evaluated at sqrt(2) / 2 pixels, and the whole map
    is point-symmetric about the array's middle. A map centered on the
    header's XCENTER instead puts one of those pixels on zero separation and
    the other three at different radii, so they disagree.
    """
    with jax.enable_x64():
        coro = axis_concentrated_coro
        pixel_lod = float(coro.pixel_scale_arcsec.value)
        intensity_map = np.asarray(
            stellar_intensity_curves(
                EqxCoronagraph(yippy_coro=coro), 0.0, interpolation="keys"
            ).evaluate(
                "core_mean_intensity", np.hypot(*coro.stellar_intens.grid.coordinates())
            )
        )
        middle = intensity_map.shape[0] // 2
        central = intensity_map[middle - 1 : middle + 1, middle - 1 : middle + 1]

        expected = float(
            stellar_intensity_curves(
                EqxCoronagraph(yippy_coro=coro), 0.0, interpolation="keys"
            ).evaluate("core_mean_intensity", np.sqrt(0.5) * pixel_lod)
        )
        np.testing.assert_allclose(central, expected, rtol=1e-10)
        np.testing.assert_allclose(intensity_map, intensity_map[::-1, ::-1], rtol=1e-10)


@pytest.mark.parametrize("npix,kind", [(8, "2dq"), (9, "2dq"), (8, "1d")])
def test_datacube_source_grid_matches_point_sources(tmp_path, npix, kind):
    """Fails if scene deltas and point sources differ by half a native pixel.

    Float32 FFT floor: 2e-6 of peak for batched/single reads and reflected
    quadrants. The explicit half-pixel mutant must differ by >0.1 of peak,
    over 50,000 times the allowed error. Odd grids retain integer offsets.
    """
    from copy import copy

    from synthetic_yip import write_synthetic_yip

    from yippy import Coronagraph

    package = write_synthetic_yip(
        tmp_path / "cube",
        kind=kind,
        npix=npix,
        max_offset_lod=1.5,
        offset_step_lod=0.25,
        psf_sigma_lod=0.25,
        occulter_floor=1.0,
    )
    full = Coronagraph(package.path, use_quarter_psf_datacube=False)
    full.create_psf_datacube(batch_size=81)
    cube = np.asarray(full.psf_datacube)
    center = (npix - 1) / 2
    row, col = npix // 2 - 1, npix // 2 + 1
    x, y = (col - center) * 0.25, (row - center) * 0.25
    direct = np.asarray(full.offax(x * lod, y * lod).data)
    scene = np.zeros((npix, npix))
    scene[row, col] = 1
    rendered = np.einsum("ij,ijyx->yx", scene, cube)
    tolerance = direct.max() * 2e-6
    np.testing.assert_allclose(rendered, direct, rtol=0, atol=tolerance)
    shifted = np.asarray(full.create_psf(x + 0.125, y + 0.125).data)
    assert np.max(np.abs(shifted - direct)) > direct.max() * 0.1
    # Independent integer-offset reader anchor for odd n, half-integer for even.
    assert x == (col - ((npix - 1) // 2)) * 0.25 if npix % 2 else x == 0.375
    quarter = copy(full)
    quarter.use_quarter_psf_datacube = True
    quarter._create_psf_datacube_gpu(batch_size=81)
    small = np.asarray(quarter.psf_datacube)
    assert small.shape[:2] == ((npix + 1) // 2, (npix + 1) // 2)
    folded = np.empty_like(cube)
    start = int(np.ceil(center))
    for j in range(npix):
        for i in range(npix):
            jj, ii = (
                (npix - 1 - j if j < center else j),
                (npix - 1 - i if i < center else i),
            )
            plane = small[jj - start, ii - start]
            if j < center:
                plane = plane[::-1, :]
            if i < center:
                plane = plane[:, ::-1]
            folded[j, i] = plane
    np.testing.assert_allclose(folded, cube, rtol=0, atol=cube.max() * 2e-6)
    quarter._datacube_cache_path.unlink()
    quarter.create_psf_datacube(batch_size=81)
    np.testing.assert_allclose(
        quarter.psf_datacube, small, rtol=0, atol=small.max() * 2e-6
    )
    assert full._datacube_identity()["layout"] == 2
    # Force the batched builder to generate the full cube rather than load it.
    full._datacube_cache_path.unlink()
    full._create_psf_datacube_gpu(batch_size=81)
    np.testing.assert_allclose(full.psf_datacube, cube, rtol=0, atol=cube.max() * 2e-6)
