"""The contract the shipped package and its reader hold each other to.

These are the assertions that can only be made against real data: the header
keywords a package is required to carry, the normalization its PSFs are written
with, and the working angles the reader derives from them. They are data-gated.
A cold cache with no network skips with a reason naming the cause, setting
``YIPPY_REQUIRE_REFERENCE_DATA=1`` turns that skip into a failure, and a cache
that is present and disagrees fails either way.

Numbers pinned here are measured properties of the shipped package rather than
derivations, so each one records what it means and how far it is allowed to
move. Where a quantity has a closed form it is tested against the synthetic
package instead, in ``test_analytic_yip.py``.
"""

import astropy.io.fits as pyfits
import numpy as np
import pytest
from lod_unit import lod

# The off-axis PSFs are normalized so that a source far outside the mask
# carries this fraction of the light an unocculted source would deliver to the
# array. It is a property of how the package was written, not of the reader,
# and the reader has no keyword to read it from, so it is recorded here until
# a future package declares it. Measured on eac1_aavc_2d.
FAR_FIELD_PSF_SUM = 0.485
FAR_FIELD_TOL = 2e-2

# The separation at which the PSF has cleared the mask but is still comfortably
# inside the array. Beyond about 28 lambda/D the blob starts falling off the
# 256-pixel edge and the sum drops for a geometric reason.
FAR_FIELD_SEPARATIONS_LOD = (20.0, 25.0)

# Inner working angle of the shipped package, pinned so that a change in the
# reader's half-maximum crossing is visible in review. This is a regression
# pin, not a derivation: it catches unintended change and establishes nothing
# about correctness.
# Measured with the source-centred 10x band-limited circle of radius 0.7 lambda/D.
# Every record uses the same PCHIP half-maximum crossing.
EAC1_AAVC_IWA_LOD = 4.054787
EAC1_AAVC_IWA_TOL = 5e-3


def test_the_header_declares_a_usable_pixel_scale(shipped_coro):
    """The reader's pixel scale is the header's, and it is physically sensible.

    Everything downstream converts separations with this number, so a package
    that omitted it or carried it in the wrong unit would move every planet in
    every simulated frame. The range is wide on purpose: it excludes a scale
    given in arcseconds or in milliarcseconds without excluding any plausible
    lambda/D sampling.
    """
    header = pyfits.getheader(shipped_coro.yip_path / "stellar_intens.fits", 0)
    declared = header["PIXSCALE"]

    assert float(shipped_coro.pixel_scale_arcsec.value) == pytest.approx(declared), (
        "the reader's pixel scale is not the one the header declares"
    )
    assert 0.05 <= declared <= 1.0, (
        f"a pixel scale of {declared} lambda/D per pixel is outside the range a "
        "yield input package is sampled at; the value is probably in another unit"
    )


def test_the_off_axis_normalization_is_the_recorded_one(shipped_coro):
    """A far-field off-axis PSF sums to the package's normalization constant.

    This is the absolute scale every consumer inherits. A shape or ratio test
    cannot see it, because a package rescaled by a constant factor passes every
    one of them, and the factor propagates into every count rate built on the
    package.
    """
    for separation in FAR_FIELD_SEPARATIONS_LOD:
        total = float(
            np.asarray(shipped_coro.offax(separation * lod, 0.0 * lod).data).sum()
        )
        assert total == pytest.approx(FAR_FIELD_PSF_SUM, rel=FAR_FIELD_TOL), (
            f"the off-axis PSF at {separation} lambda/D sums to {total:.6f}, not "
            f"the recorded {FAR_FIELD_PSF_SUM} normalization"
        )


def test_the_far_field_normalization_has_actually_plateaued(shipped_coro):
    """The recorded constant is a plateau, not a point on a rising curve.

    Pinning a number from a curve that is still climbing would make the
    tolerance above meaningless, since any separation would give a different
    answer. Checking that two well-separated far-field points agree is what
    makes the constant a property of the package.
    """
    sums = [
        float(np.asarray(shipped_coro.offax(s * lod, 0.0 * lod).data).sum())
        for s in FAR_FIELD_SEPARATIONS_LOD
    ]
    spread = (max(sums) - min(sums)) / np.mean(sums)
    assert spread < FAR_FIELD_TOL, (
        f"the far-field PSF sums {sums} still vary by {spread:.3f} across the "
        "separations the normalization is read from, so it has not plateaued"
    )


def test_inscribed_conversion_is_explicit(syn_coro_1d):
    """Fails if the removed flag is accepted or the old D/D_INSC factor changes.

    Exact-algebra: typed header values D=6 m, D_INSC=5 m give 1.2.
    """
    from yippy import Coronagraph, EqxCoronagraph, lod_inscribed_to_circumscribed

    coro = syn_coro_1d
    assert coro.header.diameter.to_value("m") == 6.0
    assert coro.header.diameter_inscribed.to_value("m") == 5.0
    for constructor, kwargs in (
        (Coronagraph, {"yip_path": coro.yip_path}),
        (EqxCoronagraph, {"yip_path": coro.yip_path}),
        (EqxCoronagraph, {"yippy_coro": coro}),
    ):
        with pytest.raises(TypeError, match="use_inscribed_diameter"):
            constructor(**kwargs, use_inscribed_diameter=True)
    for separation in (5.0, 10.0):
        assert (
            lod_inscribed_to_circumscribed(separation, coro.grid.frame, 5.0)
            == separation * 1.2
        )


@pytest.mark.parametrize(
    "centre,expected", [("source", EAC1_AAVC_IWA_LOD), ("peak", 4.060305)]
)
def test_the_inner_working_angle_has_not_moved(shipped_coro, centre, expected):
    """A regression pin on the derived inner working angle.

    Every yield number depends on where the reader says the mask opens. This
    catches an unintended change in the half-maximum crossing or in the
    throughput curve beneath it; it is not evidence that the value is right.
    The closed-form check of the same derivation is in ``test_analytic_yip.py``.
    """
    from yippy import Circular, performance_curves

    record = (
        shipped_coro.performance
        if centre == "source"
        else performance_curves(
            shipped_coro,
            shipped_coro.stellar_intens(0.0 * lod),
            Circular(0.7, centre=centre),
        )
    )
    assert record.rule.centre == centre
    got = record.iwa_lod
    assert got == pytest.approx(expected, abs=EAC1_AAVC_IWA_TOL), (
        f"the inner working angle moved to {got:.4f} lambda/D "
        f"from the pinned {expected}"
    )


def test_data_extent_is_the_array_half_width(shipped_coro):
    """Fails if the array half-width is mislabeled as an outer working angle."""
    expected = min(shipped_coro.grid.shape) / 2 * shipped_coro.grid.step
    assert shipped_coro.data_extent_lod == pytest.approx(expected)
    assert not hasattr(shipped_coro, "OWA")
    assert shipped_coro.data_extent_lod > shipped_coro.iwa_lod


def test_data_extent_does_not_claim_a_dark_hole_boundary(shipped_coro):
    """Fails if the array edge again implies a usable low-contrast region.

    The shipped stellar map rises toward the array edge. The previous OWA
    sentinel xfailed because a geometric bound promised optical performance;
    the new API makes that distinction explicit and retains this physical
    counterexample. A factor of ten is deliberately below the measured rise.
    """
    extent = shipped_coro.data_extent_lod
    reference = max(
        float(shipped_coro.raw_contrast(s))
        for s in np.linspace(shipped_coro.iwa_lod + 1.0, 0.75 * extent, 8)
    )
    probes = np.linspace(0.75 * extent, extent, 9)
    values = np.asarray(shipped_coro.raw_contrast(probes))
    assert np.all(np.isfinite(values))
    assert values.max() > 10.0 * reference
    assert not hasattr(shipped_coro, "OWA")


def test_header_grid_corner_counts(syn_coro_1d):
    """Fails on corner/center confusion or swapped rectangular axes.

    Exact binary quarter-pixel arithmetic needs zero tolerance.
    """
    from dataclasses import replace

    header = replace(
        syn_coro_1d.header, naxis1=256, naxis2=256, xcenter=128, ycenter=128
    )
    grid = header.pixel_grid()
    assert (grid.center_x, grid.center_y) == (127.5, 127.5)
    assert grid.pixel_to_unit(128, 127) == (0.125, -0.125)
    missing = replace(header, xcenter=None, ycenter=None).pixel_grid((9, 12))
    assert (missing.center_x, missing.center_y) == (5.5, 4.0)
    assert grid.frame.reference_wavelength_nm == 500.0
    assert grid.frame.diameter_m == 6.0
    assert grid.frame.diameter_convention == "circumscribed"


def test_public_images_carry_header_grid(syn_coro_1d, syn_eqx_1d, monkeypatch):
    """Fails if a public image drops geometry, reconstruction, measure or axes.

    Metadata are exact; the node image is checked at the float32 FFT floor
    (2e-7 of peak).
    """
    from multiprocessing.pool import ThreadPool

    import equinox as eqx
    import jax.numpy as jnp
    from hwoutils.grids import Image

    from yippy.offax import OffAx

    coro, view = syn_coro_1d, syn_eqx_1d
    grid = coro.header.pixel_grid()
    numpy_reader = OffAx(
        coro.yip_path,
        "offax_psf.fits",
        "offax_psf_offset_list.fits",
        coro.header.pixscale,
        True,
        True,
    )
    for product in (
        coro,
        coro.offax,
        numpy_reader,
        coro.stellar_intens,
        coro.sky_trans,
        view,
    ):
        assert product.grid == grid
        assert not hasattr(product, "center_x")
        assert not hasattr(product, "center_y")
    with pytest.warns(DeprecationWarning):
        numpy_single = numpy_reader.create_psf(1.0, 0.0)
    with pytest.warns(DeprecationWarning):
        numpy_stack = numpy_reader.create_psfs([1.0], [0.0])
    # Exercise the cube product without charging process startup to metadata.
    monkeypatch.setattr("yippy.offax.Pool", ThreadPool)
    with pytest.warns(DeprecationWarning):
        numpy_cube = numpy_reader.create_psfs_parallel(
            np.array([1.0]), np.array([0.0]), workers=1
        )
    images = [
        numpy_single,
        numpy_stack,
        numpy_cube,
        numpy_reader.get_psf_by_offset_idx(2, 0),
        coro.create_psf(1.0, 0.0),
        coro.offax(1.0 * lod, 0.0 * lod),
        coro.offax.get_psf_by_offset_idx(2, 0),
        coro.create_psfs(jnp.array([1.0]), jnp.array([0.0])),
        coro.offax.create_psfs_parallel(jnp.array([1.0]), jnp.array([0.0])),
        coro.stellar_intens(0.1 * lod),
        coro.stellar_intens(np.array([0.0, 0.1]) * lod),
        eqx.filter_jit(view.create_psf)(1.0, 0.0),
        view.create_psfs(jnp.array([1.0]), jnp.array([0.0])),
        eqx.filter_jit(view.stellar_intens)(0.1),
        view.stellar_intens(jnp.array([0.0, 0.1])),
        view.source_psfs(
            500,
            jnp.array([1.0]),
            jnp.array([0.0]),
            pixel_scale_lod=0.25,
            shape=grid.shape,
        ),
        view.stellar_map(500, 0.1, pixel_scale_lod=0.25, shape=grid.shape),
    ]
    for image in images:
        assert isinstance(image, Image)
        assert image.grid == grid
        assert image.data.shape[-2:] == grid.shape
        assert len(image.leading_axes) == image.data.ndim - 2
        assert (image.measure, image.quantity) == ("per_pixel", "stellar_flux_fraction")
        assert image.reconstruction == "band_limited"
    for image in (
        coro.sky_trans(),
        view.sky_trans,
        view.background_transmission(500, pixel_scale_lod=0.25, shape=grid.shape),
    ):
        assert isinstance(image, Image) and image.grid == grid
        assert image.data.shape == grid.shape
        assert (image.measure, image.quantity) == ("per_unit_area", "transmission")
        assert image.reconstruction == "linear"
        assert image.outside == "reflect"
    # Stellar halos continue past the field; off-axis PSFs vanish outside it.
    for image in (
        coro.stellar_intens(0.1 * lod),
        view.stellar_intens(jnp.array([0.0, 0.1])),
        view.stellar_map(500, 0.1, pixel_scale_lod=0.25, shape=grid.shape),
    ):
        assert image.outside == "reflect"
    for image in (
        coro.create_psf(1.0, 0.0),
        view.create_psfs(jnp.array([1.0]), jnp.array([0.0])),
    ):
        assert image.outside == "zero"
    # A coarser detector cannot promise to reconstruct the native bandwidth.
    for image in (
        view.source_psfs(
            500,
            jnp.array([1.0]),
            jnp.array([0.0]),
            pixel_scale_lod=0.5,
            shape=(32, 32),
        ),
        view.stellar_map(500, 0.1, pixel_scale_lod=0.5, shape=(32, 32)),
    ):
        assert image.reconstruction == "linear"
    np.testing.assert_allclose(
        numpy_single.data,
        coro.create_psf(1.0, 0.0).data,
        rtol=0,
        atol=float(numpy_single.data.max()) * 2e-7,
    )


def test_detector_aperture_converges_to_performance(syn_coro_1d, syn_eqx_1d):
    """Fails if detector rendering silently uses linear PSF reconstruction.

    Solution verification: refine detector pitch by two at fixed angular extent.
    Pixel-box integration and fractional aperture coverage give second-order
    error against the performance code's subpixel intensity integral. Spike
    basis: the synthetic Gaussian's error/pitch**2 is below 0.16; allow 0.18.
    The observed-order band permits aperture-edge phase and reference quadrature
    error, while rejecting the nonvanishing linear-reconstruction bias.
    """
    from dataclasses import replace

    import jax

    from yippy import Circular, performance_curves

    with jax.enable_x64():
        rule, separation = Circular(0.7), 2.0
        record = performance_curves(
            syn_coro_1d,
            syn_eqx_1d.stellar_intens(0.0),
            rule,
            separations_lod=[separation],
        )
        expected = record.evaluate("throughput", separation)
        errors, sums = [], []
        pitches = (0.24, 0.12, 0.06)
        for pitch, size in zip(pitches, (32, 64, 128), strict=True):
            stack = syn_eqx_1d.source_psfs(
                500.0,
                np.array([separation]),
                np.array([0.0]),
                pixel_scale_lod=pitch,
                shape=(size, size),
            )
            psf = replace(stack, data=stack.data[0], leading_axes=())
            assert psf.reconstruction == "band_limited"
            weights = rule.weights(psf, (separation, 0.0))
            sums.append(float(np.sum(np.asarray(psf.data) * np.asarray(weights))))
            errors.append(abs(sums[-1] - expected))
            assert errors[-1] < 0.18 * pitch**2
        orders = np.log2(np.array(errors[:-1]) / errors[1:])
        assert np.all((orders > 1.7) & (orders < 2.3))
        print(f"detector sums={sums}; throughput={expected}; errors={errors}")
        print(f"observed orders={orders.tolist()}")


def test_public_package_optical_axis_evidence(shipped_coro):
    """Fails if the public package's corner count is treated as a pixel index.

    Spike basis: flipped stellar relative RMS is about 2e-6 at 127.5,
    versus 0.37 at 127 or 128. Bound 1e-5 and ratio 1e4 retain margin.
    A three-sample log-parabola locates the far-field peak within 0.1 pixel;
    the wrong center is 0.5 pixel away, five times that interpolation bound.
    """
    coro = shipped_coro
    assert (coro.grid.center_x, coro.grid.center_y) == (127.5, 127.5)
    image = coro.stellar_intens.psfs[0]

    def rms(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(a)

    correct = rms(image, image[::-1, ::-1])
    wrong = [
        rms(image[:-1, :-1], image[:-1, :-1][::-1, ::-1]),
        rms(image[1:, 1:], image[1:, 1:][::-1, ::-1]),
    ]
    assert correct < 1e-5
    assert min(wrong) > 1e4 * correct
    for separation in (5.0, 10.0, 20.0):
        psf = np.asarray(coro.create_psf(separation, 0.0).data)
        row, col = np.unravel_index(np.argmax(psf), psf.shape)
        a, b, c = np.log(psf[row, col - 1 : col + 2])
        peak_x = col + 0.5 * (a - c) / (a - 2 * b + c)
        assert abs(peak_x - (127.5 + separation / 0.25)) < 0.1
