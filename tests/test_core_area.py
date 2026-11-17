"""The core area and throughput are those of the PSF the image simulator renders.

With a PSF truncation ratio ``r`` the core of an off-axis PSF is the region
where its intensity exceeds ``r`` times its peak. The record
``core_area_lod2`` is that region's solid angle in (lambda/D)^2, the aperture an
exposure-time calculator multiplies the leaked starlight and the background by,
and ``PerformanceCurves.throughput`` is the fraction of the source's light inside it,
which sets the planet count rate. At every separation, on or between the
tabulated offsets, the default curves interpolate the measurements of the
package PSFs. Explicit separation steps opt into synthesized-PSF measurements.

The references here do not come from the code under test. A circular Gaussian
of standard deviation sigma and flux F has the core area
``2 pi sigma^2 ln(1 / r)`` and the core throughput ``F (1 - r)``. Between
two tabulated offsets yippy synthesizes the PSF as the two neighboring PSFs
shifted onto the source and mixed with weights linear in separation; for
concentric Gaussians that mixture is radially symmetric and its core radius is
the root of a one-dimensional equation. On a shipped package the reference is
the synthesized PSF's band-limited reconstruction, thresholded and restricted
to the peak-connected region.
"""

import astropy.io.fits as pyfits
import numpy as np
import pytest
from conftest import REFERENCE_DATA_REQUIRED
from optixstuff.apertures import HalfMaximum
from optixstuff.performance import performance_curves
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from yippy import Coronagraph
from yippy.performance import CORE_STEP_LOD, core_separations

# Separations (lambda/D) probed, on and between the tabulated offsets.
GAUSSIAN_SEPARATIONS_LOD = (2.1, 2.37, 3.0, 3.62, 4.13, 5.05)
TRUNC_RATIOS = (0.3, 0.5)

# Offset grids of the Gaussian package: the default one puts every PSF center
# on a pixel center (0.5 lambda/D steps on 0.25 lambda/D pixels), the other
# puts them at fractions of a pixel.
OFFSET_GRIDS = {"on_pixels": (0.0, 0.5), "off_pixels": (0.07, 0.37)}

# Relative tolerance on the core area of a smooth, well-sampled PSF. The area
# is counted in cells a tenth of a pixel (0.025 lambda/D) on a side; a core of
# radius 0.6 lambda/D centered on a pixel, where whole rings of symmetric
# cells cross the threshold together, sits within 0.8 percent of the closed
# form, and off-center cores sit closer.
AREA_TOL = 1e-2

# Relative tolerance on the core throughput. A lattice error in the area moves
# the throughput by the light in the boundary cells, which sit at the threshold
# intensity, so the throughput error is r ln(1/r) / (1 - r) times the area
# error: about 0.7 of it at r = 0.5.
THROUGHPUT_TOL = 1e-2

# A package whose PSF width grows with separation, tabulated every 2 lambda/D,
# so a source between offsets gets a PSF whose core is neither neighbor's.
GROWTH_OFFSET_STEP_LOD = 2.0
GROWTH_SIGMA0_LOD = 0.35
GROWTH_SIGMA_SLOPE = 0.1


def gaussian_core_area(sigma_lod, ratio):
    """Core area of a circular Gaussian above ``ratio`` times its peak."""
    return 2.0 * np.pi * sigma_lod**2 * np.log(1.0 / ratio)


@pytest.fixture(scope="module", params=sorted(OFFSET_GRIDS))
def gaussian_package(request, tmp_path_factory):
    """The radially symmetric analytic package: identical Gaussians."""
    from synthetic_yip import write_synthetic_yip

    first, step = OFFSET_GRIDS[request.param]
    return write_synthetic_yip(
        tmp_path_factory.mktemp("core") / request.param,
        min_offset_lod=first,
        offset_step_lod=step,
    )


@pytest.mark.parametrize("ratio", TRUNC_RATIOS)
def test_core_area_of_a_gaussian_is_its_closed_form(gaussian_package, ratio):
    """The core area of a Gaussian PSF is its closed form wherever it sits.

    Counting whole pixels above the threshold, even on a twice-oversampled
    grid, puts the area on a lattice of (0.125 lambda/D)^2 cells; for a core
    centered on a pixel the cells cross the threshold in symmetric rings and
    the area is off by 9 percent at a ratio of 0.5.
    """
    coro = Coronagraph(gaussian_package.path, rules=(HalfMaximum(ratio),))
    expected = gaussian_core_area(gaussian_package.psf_sigma_lod, ratio)
    got = np.array(
        [
            coro.curves(HalfMaximum(ratio)).evaluate("core_area_lod2", s)
            for s in GAUSSIAN_SEPARATIONS_LOD
        ]
    )
    err = np.abs(got / expected - 1.0)
    assert err.max() <= AREA_TOL, (
        f"core area departs from 2 pi sigma^2 ln(1/r) = {expected:.5f} by "
        f"{err.max():.4f} at {GAUSSIAN_SEPARATIONS_LOD[int(np.argmax(err))]} "
        f"lambda/D: {np.round(got, 5)}"
    )


def growth_sigma(separation_lod):
    """PSF standard deviation of the growth package at a tabulated offset."""
    return GROWTH_SIGMA0_LOD + GROWTH_SIGMA_SLOPE * separation_lod


@pytest.fixture(scope="module")
def growth_package(tmp_path_factory):
    """A package whose Gaussian PSFs widen with separation."""
    from synthetic_yip import _gaussian_psf, write_synthetic_yip

    package = write_synthetic_yip(
        tmp_path_factory.mktemp("core") / "growth",
        offset_step_lod=GROWTH_OFFSET_STEP_LOD,
    )
    with pyfits.open(package.path / "offax_psf.fits") as hdul:
        header = hdul[0].header.copy()
    psfs = np.stack(
        [
            _gaussian_psf(
                package.npix,
                package.pixscale,
                package.center,
                x,
                y,
                growth_sigma(np.hypot(x, y)),
                package.psf_flux * package.occulter(np.hypot(x, y)),
            )
            for x, y in package.offsets
        ]
    )
    pyfits.PrimaryHDU(psfs, header=header).writeto(
        package.path / "offax_psf.fits", overwrite=True
    )
    return package


def mixture_core(package, separation_lod, ratio):
    """Core area and throughput of the PSF yippy synthesizes between two offsets.

    The two neighboring PSFs are shifted onto the source and mixed with
    weights ``1 - u`` and ``u``, ``u`` the fractional position between them,
    so the PSF is a sum of two concentric Gaussians of known widths and
    fluxes, and the light inside a radius is a sum of closed forms.
    """
    nodes = np.sort(package.offsets[:, 0])
    lo = nodes[nodes <= separation_lod].max()
    hi = nodes[nodes > separation_lod].min()
    u = (separation_lod - lo) / (hi - lo)
    terms = []
    for node, weight in ((lo, 1.0 - u), (hi, u)):
        sigma = growth_sigma(node)
        flux = weight * package.psf_flux * package.occulter(node)
        terms.append((flux, sigma))

    def profile(rho):
        return sum(
            f / (2.0 * np.pi * s**2) * np.exp(-(rho**2) / (2.0 * s**2))
            for f, s in terms
        )

    radius = brentq(lambda rho: profile(rho) - ratio * profile(0.0), 0.0, 10.0)
    throughput = sum(f * (1.0 - np.exp(-(radius**2) / (2.0 * s**2))) for f, s in terms)
    return np.pi * radius**2, throughput


def mixture_core_area(package, separation_lod, ratio):
    """Core area of the PSF yippy synthesizes between two Gaussian offsets."""
    return mixture_core(package, separation_lod, ratio)[0]


@pytest.mark.parametrize("ratio", TRUNC_RATIOS)
def test_core_area_follows_the_synthesized_psf_between_offsets(growth_package, ratio):
    """Between tabulated offsets the core area is that of the rendered PSF.

    Interpolating the tabulated core areas linearly in separation instead
    overstates it by 4 to 9 percent here, because the core of the mixed PSF
    is not the mix of the two cores.
    """
    coro = Coronagraph(growth_package.path, rules=(HalfMaximum(ratio),))
    record = performance_curves(
        lambda s: coro.off_axis_psf(None, s, grid=None),
        coro.stellar_map(None, 0.0, grid=None),
        HalfMaximum(ratio),
        separations_lod=np.union1d(
            core_separations(coro, CORE_STEP_LOD),
            np.asarray(coro._offaxis_backend.x_offsets),
        ),
    )
    separations = np.array([2.5, 3.0, 3.37, 3.5, 4.6, 4.9])
    expected = np.array(
        [mixture_core_area(growth_package, s, ratio) for s in separations]
    )
    got = np.array([record.evaluate("core_area_lod2", s) for s in separations])
    err = np.abs(got / expected - 1.0)
    assert err.max() <= AREA_TOL, (
        f"core area departs from the synthesized PSF's by {err.max():.4f} at "
        f"{separations[int(np.argmax(err))]} lambda/D: got {np.round(got, 4)}, "
        f"expected {np.round(expected, 4)}"
    )


def fourier_upsampled_core(psf, pixel_lod, ratio, factor):
    """Core area and throughput of a PSF from its band-limited upsampling.

    A sampled PSF at or above Nyquist determines the continuous intensity,
    and zero-padding its spectrum evaluates that intensity on a grid
    ``factor`` times finer, so counting (and summing) the finer cells above
    threshold converges to the continuous core area (and the light inside
    it) without a spatial kernel.
    """
    psf = np.asarray(psf, dtype=np.float64)
    n = psf.shape[0]
    big = n * factor
    spec = np.fft.fftshift(np.fft.fft2(psf))
    padded = np.zeros((big, big), dtype=complex)
    start = (big - n) // 2
    padded[start : start + n, start : start + n] = spec
    fine = np.real(np.fft.ifft2(np.fft.ifftshift(padded)))
    above = fine > ratio * fine.max()
    connected = np.zeros_like(above)
    pending = [np.unravel_index(np.argmax(fine), fine.shape)]
    while pending:
        row, col = pending.pop()
        if connected[row, col] or not above[row, col]:
            continue
        connected[row, col] = True
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            y, x = row + dy, col + dx
            if 0 <= y < big and 0 <= x < big and above[y, x] and not connected[y, x]:
                pending.append((y, x))
    above = connected
    # The inverse transform of the padded spectrum carries 1 / factor^2, so
    # each fine cell already holds its share of the pixel's light.
    return above.sum() * (pixel_lod / factor) ** 2, fine[above].sum()


SHIPPED_RATIO = 0.3


@pytest.fixture(scope="module")
def shipped_trunc_coro():
    """The shipped AAVC package at truncation ratio 0.3, or a skip that says why."""
    from yippy import fetch_yip

    try:
        yip_path = fetch_yip("eac1_aavc_2d")
    except Exception as exc:
        message = (
            f"data-absent: eac1_aavc_2d is not cached and could not be fetched ({exc})"
        )
        if REFERENCE_DATA_REQUIRED:
            pytest.fail(message)
        pytest.skip(message)
    return Coronagraph(yip_path, rules=(HalfMaximum(SHIPPED_RATIO),))


@pytest.mark.parametrize("separation_lod", (3.3, 4.7, 6.1, 9.4))
def test_shipped_core_area_matches_the_band_limited_psf(
    shipped_trunc_coro, separation_lod
):
    """The default area curve interpolates native band-limited core measurements.

    The reference counts above-threshold cells of the native PSFs'
    band-limited reconstruction over the whole image, eight times finer than
    the package pixels, and interpolates those measurements with PCHIP.
    """
    coro = shipped_trunc_coro
    pixel_lod = float(coro.grid.step)
    y_offsets = np.asarray(coro._offaxis_backend.y_offsets, dtype=float)
    nodes = np.asarray(coro._offaxis_backend.x_offsets)
    valid = [
        i
        for i, node in enumerate(nodes)
        if node >= 0
        and coro._offaxis_backend.get_psf_by_offset_idx(
            i, int(np.argmin(np.abs(y_offsets)))
        )
        is not None
    ]
    native_nodes = nodes[valid]
    i = np.searchsorted(native_nodes, separation_lod, side="right") - 1
    # PCHIP needs the neighboring secants to set both interval derivatives.
    stencil = native_nodes[max(0, i - 1) : min(len(native_nodes), i + 3)]
    values = [
        fourier_upsampled_core(
            coro._offaxis_backend.get_psf_by_offset_idx(
                int(np.flatnonzero(nodes == s)[0]), int(np.argmin(np.abs(y_offsets)))
            ).data,
            pixel_lod,
            SHIPPED_RATIO,
            factor=8,
        )[0]
        for s in stencil
    ]
    expected = PchipInterpolator(stencil, values)(separation_lod)
    got = coro.curves(HalfMaximum(SHIPPED_RATIO)).evaluate(
        "core_area_lod2", separation_lod
    )
    assert got == pytest.approx(expected, rel=AREA_TOL)


@pytest.mark.parametrize("ratio", TRUNC_RATIOS)
def test_core_throughput_of_a_gaussian_is_its_closed_form(gaussian_package, ratio):
    """The core throughput of a Gaussian PSF is ``(1 - r)`` times its flux.

    yippy mixes the two neighboring PSFs with weights linear in separation,
    so the flux of the PSF it synthesizes is the neighbors' fluxes
    interpolated linearly. Summing the light in whole twice-oversampled
    pixels misplaces the core boundary and is off by several percent for a
    core centered on a pixel.
    """
    coro = Coronagraph(gaussian_package.path, rules=(HalfMaximum(ratio),))
    nodes = np.sort(gaussian_package.offsets[:, 0])
    flux = np.interp(
        GAUSSIAN_SEPARATIONS_LOD,
        nodes,
        gaussian_package.psf_flux * gaussian_package.occulter(nodes),
    )
    expected = flux * (1.0 - ratio)
    got = np.array(
        [
            coro.curves(HalfMaximum(ratio)).evaluate("throughput", s)
            for s in GAUSSIAN_SEPARATIONS_LOD
        ]
    )
    err = np.abs(got / expected - 1.0)
    assert err.max() <= THROUGHPUT_TOL, (
        f"core throughput departs from F (1 - r) by {err.max():.4f} at "
        f"{GAUSSIAN_SEPARATIONS_LOD[int(np.argmax(err))]} lambda/D: got "
        f"{np.round(got, 5)}, expected {np.round(expected, 5)}"
    )


@pytest.mark.parametrize("ratio", TRUNC_RATIOS)
def test_core_throughput_follows_the_synthesized_psf_between_offsets(
    growth_package, ratio
):
    """Between tabulated offsets the core throughput is that of the rendered PSF.

    Interpolating the tabulated throughputs linearly in separation instead
    overstates it by 2 to 7 percent here.
    """
    coro = Coronagraph(growth_package.path, rules=(HalfMaximum(ratio),))
    record = performance_curves(
        lambda s: coro.off_axis_psf(None, s, grid=None),
        coro.stellar_map(None, 0.0, grid=None),
        HalfMaximum(ratio),
        separations_lod=np.union1d(
            core_separations(coro, CORE_STEP_LOD),
            np.asarray(coro._offaxis_backend.x_offsets),
        ),
    )
    separations = np.array([2.5, 3.0, 3.37, 3.5, 4.6, 4.9])
    expected = np.array(
        [mixture_core(growth_package, s, ratio)[1] for s in separations]
    )
    got = np.array([record.evaluate("throughput", s) for s in separations])
    err = np.abs(got / expected - 1.0)
    assert err.max() <= THROUGHPUT_TOL, (
        f"core throughput departs from the synthesized PSF's by {err.max():.4f} "
        f"at {separations[int(np.argmax(err))]} lambda/D: got {np.round(got, 4)}, "
        f"expected {np.round(expected, 4)}"
    )


@pytest.mark.parametrize("separation_lod", (3.3, 4.7, 6.1, 9.4))
def test_shipped_core_throughput_matches_the_band_limited_psf(
    shipped_trunc_coro, separation_lod
):
    """The default throughput curve interpolates native band-limited core fluxes."""
    coro = shipped_trunc_coro
    pixel_lod = float(coro.grid.step)
    y_offsets = np.asarray(coro._offaxis_backend.y_offsets, dtype=float)
    nodes = np.asarray(coro._offaxis_backend.x_offsets)
    valid = [
        i
        for i, node in enumerate(nodes)
        if node >= 0
        and coro._offaxis_backend.get_psf_by_offset_idx(
            i, int(np.argmin(np.abs(y_offsets)))
        )
        is not None
    ]
    native_nodes = nodes[valid]
    i = np.searchsorted(native_nodes, separation_lod, side="right") - 1
    # PCHIP needs the neighboring secants to set both interval derivatives.
    stencil = native_nodes[max(0, i - 1) : min(len(native_nodes), i + 3)]
    values = [
        fourier_upsampled_core(
            coro._offaxis_backend.get_psf_by_offset_idx(
                int(np.flatnonzero(nodes == s)[0]), int(np.argmin(np.abs(y_offsets)))
            ).data,
            pixel_lod,
            SHIPPED_RATIO,
            factor=8,
        )[1]
        for s in stencil
    ]
    expected = PchipInterpolator(stencil, values)(separation_lod)
    got = coro.curves(HalfMaximum(SHIPPED_RATIO)).evaluate("throughput", separation_lod)
    assert got == pytest.approx(expected, rel=THROUGHPUT_TOL)


# A Gaussian package tabulated out to 7.5 lambda/D on a 64-pixel (16 lambda/D)
# image, so offsets reach within half a lambda/D of the edge.
EDGE_MAX_OFFSET_LOD = 7.5


@pytest.fixture(scope="module")
def edge_package(tmp_path_factory):
    """Identical Gaussians tabulated up to the image edge."""
    from synthetic_yip import write_synthetic_yip

    return write_synthetic_yip(
        tmp_path_factory.mktemp("core") / "edge", max_offset_lod=EDGE_MAX_OFFSET_LOD
    )


def gaussian_core_expected(package, separations, ratio):
    """PCHIP of closed-form native Gaussian fluxes, and their constant core area."""
    nodes = np.sort(package.offsets[:, 0])
    flux = PchipInterpolator(nodes, package.psf_flux * package.occulter(nodes))(
        separations
    )
    return flux * (1.0 - ratio), gaussian_core_area(package.psf_sigma_lod, ratio)


@pytest.mark.parametrize("ratio", TRUNC_RATIOS)
def test_core_near_the_image_edge_is_its_closed_form(edge_package, ratio):
    """A core that lies inside the image is measured whole near the edge.

    The sources sit 1 to 2 lambda/D from the edge, where the core (radius
    0.6 to 0.8 lambda/D) is still inside the image; a band-limited
    interpolant evaluated next to the edge would wrap light from the
    opposite border into it.
    """
    coro = Coronagraph(edge_package.path, rules=(HalfMaximum(ratio),))
    separations = (6.0, 6.5, 7.0)
    t_expected, a_expected = gaussian_core_expected(edge_package, separations, ratio)
    t_got = np.array(
        [coro.curves(HalfMaximum(ratio)).evaluate("throughput", s) for s in separations]
    )
    a_got = np.array(
        [
            coro.curves(HalfMaximum(ratio)).evaluate("core_area_lod2", s)
            for s in separations
        ]
    )
    assert np.abs(a_got / a_expected - 1.0).max() <= AREA_TOL, a_got
    assert np.abs(t_got / t_expected - 1.0).max() <= THROUGHPUT_TOL, t_got


@pytest.mark.parametrize("ratio", TRUNC_RATIOS)
def test_core_inside_the_inner_working_angle_is_its_closed_form(
    gaussian_package, ratio
):
    """Inside the inner working angle a faint compact core is still measured.

    The occulter dims the synthetic package's PSFs by up to six orders of
    magnitude near the axis without changing their shape, so the core is
    defined relative to the PSF's own peak and keeps its closed form.
    """
    coro = Coronagraph(gaussian_package.path, rules=(HalfMaximum(ratio),))
    separations = (0.1, 0.25, 0.5, 0.8)
    t_expected, a_expected = gaussian_core_expected(
        gaussian_package, separations, ratio
    )
    t_got = np.array(
        [coro.curves(HalfMaximum(ratio)).evaluate("throughput", s) for s in separations]
    )
    a_got = np.array(
        [
            coro.curves(HalfMaximum(ratio)).evaluate("core_area_lod2", s)
            for s in separations
        ]
    )
    assert np.abs(a_got / a_expected - 1.0).max() <= AREA_TOL, a_got
    assert np.abs(t_got / t_expected - 1.0).max() <= THROUGHPUT_TOL, t_got


def quintic_spline_core(psf, pixel_lod, ratio, factor=40):
    """Core (throughput, area) from a quintic spline of the PSF, not a DFT.

    The spline (scipy ``RectBivariateSpline``, degree 5 in each axis) is
    evaluated at subpixel centers over the window
    :func:`optixstuff.performance.truncation_core` uses, a reference built with a
    different interpolant. The 40x reference suppresses its own whole-cell
    boundary error below the comparison tolerance.
    """
    from optixstuff.performance import core_window_half_width_pix
    from scipy.interpolate import RectBivariateSpline

    psf = np.asarray(psf, dtype=np.float64)
    n = psf.shape[0]
    idx = np.arange(n)
    spline = RectBivariateSpline(idx, idx, psf, kx=5, ky=5)
    row, col = np.unravel_index(np.argmax(psf), psf.shape)
    half = core_window_half_width_pix(pixel_lod)
    r0, r1 = max(row - half, 0), min(row + half + 1, n)
    c0, c1 = max(col - half, 0), min(col + half + 1, n)
    rows = r0 + (np.arange((r1 - r0) * factor) + 0.5) / factor - 0.5
    cols = c0 + (np.arange((c1 - c0) * factor) + 0.5) / factor - 0.5
    fine = spline(rows, cols)
    core = fine > ratio * fine.max()
    return fine[core].sum() / factor**2, core.sum() * (pixel_lod / factor) ** 2


SHIPPED_PROBES_LOD = (2.0, 4.7, 9.4, 15.0, 25.0)


@pytest.mark.parametrize("separation_lod", SHIPPED_PROBES_LOD)
def test_shipped_core_matches_a_quintic_spline_reference(
    shipped_trunc_coro, separation_lod
):
    """The band-limited core agrees with one taken from a quintic spline.

    Both interpolants reconstruct a well-sampled PSF to far better than the
    threshold needs, so the cores of the same synthesized PSF agree to 0.2
    percent; a cubic kernel does not (it overstates the area by up to 0.7
    percent at this sampling).
    """
    import jax.numpy as jnp
    from optixstuff.performance import core_window_half_width_pix, truncation_core

    coro = shipped_trunc_coro
    pixel_lod = float(coro.grid.step)
    psf = coro.off_axis_psf(None, separation_lod, grid=None).data
    t_ref, a_ref = quintic_spline_core(psf, pixel_lod, SHIPPED_RATIO)
    t_got, a_got = truncation_core(
        jnp.asarray(psf),
        SHIPPED_RATIO,
        pixel_lod,
        half_width_pix=core_window_half_width_pix(pixel_lod),
    )
    assert float(a_got) == pytest.approx(a_ref, rel=2e-3)
    assert float(t_got) == pytest.approx(t_ref, rel=2e-3)


@pytest.mark.parametrize("separation_lod", SHIPPED_PROBES_LOD)
def test_core_converges_in_subpixels(shipped_trunc_coro, separation_lod):
    """Ten subpixels per side agree with twenty to half a percent."""
    import jax.numpy as jnp
    from optixstuff.performance import core_window_half_width_pix, truncation_core

    coro = shipped_trunc_coro
    pixel_lod = float(coro.grid.step)
    psf = jnp.asarray(coro.off_axis_psf(None, separation_lod, grid=None).data)
    half = core_window_half_width_pix(pixel_lod)
    t10, a10 = truncation_core(
        psf, SHIPPED_RATIO, pixel_lod, oversample=10, half_width_pix=half
    )
    t20, a20 = truncation_core(
        psf, SHIPPED_RATIO, pixel_lod, oversample=20, half_width_pix=half
    )
    assert float(a10) == pytest.approx(float(a20), rel=5e-3)
    assert float(t10) == pytest.approx(float(t20), rel=5e-3)


def test_default_core_separations_are_the_package_offsets(growth_package):
    """Default curves rely only on PSFs supplied by the package."""
    rule = HalfMaximum(0.5)
    coro = Coronagraph(growth_package.path, rules=(rule,))
    sep = coro.curves(rule).separation_lod
    nodes = np.unique(growth_package.offsets[:, 0])
    np.testing.assert_array_equal(sep, nodes[nodes >= 0])


def test_synthesized_core_separations_remain_opt_in(growth_package):
    """Explicit synthesis adds the quarter-lambda/D grid between native offsets."""
    coro = Coronagraph(growth_package.path)
    record = performance_curves(
        lambda s: coro.off_axis_psf(None, s, grid=None),
        coro.stellar_map(None, 0.0, grid=None),
        HalfMaximum(0.5),
        separations_lod=np.union1d(
            core_separations(coro, CORE_STEP_LOD),
            coro.curves().separation_lod,
        ),
    )
    sep = record.separation_lod
    np.testing.assert_array_equal(sep, np.append(np.arange(0, 5.01, 0.25), 6.0))
