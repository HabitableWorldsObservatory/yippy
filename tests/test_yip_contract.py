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
from optixstuff.grids import Measure, PixelGrid

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
    header = pyfits.getheader(
        shipped_coro.package.source.path / "stellar_intens.fits", 0
    )
    declared = header["PIXSCALE"]

    assert float(shipped_coro.grid.step) == pytest.approx(declared), (
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
            np.asarray(
                shipped_coro._offaxis_backend(separation * lod, 0.0 * lod)
                .integrated()
                .data
            ).sum()
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
        float(
            np.asarray(
                shipped_coro._offaxis_backend(s * lod, 0.0 * lod).integrated().data
            ).sum()
        )
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
    from yippy import lod_inscribed_to_circumscribed

    coro = syn_coro_1d
    assert coro.header.diameter.to_value("m") == 6.0
    assert coro.header.diameter_inscribed.to_value("m") == 5.0
    with pytest.raises(TypeError, match="use_inscribed_diameter"):
        coro.package.prepare(use_inscribed_diameter=True)
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
    from optixstuff.apertures import Circular

    rule = Circular(0.7, centre=centre)
    record = (
        shipped_coro.curves()
        if centre == "source"
        else shipped_coro.with_rules((rule,)).curves(rule)
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
    assert shipped_coro.curves().data_extent_lod == pytest.approx(expected)
    assert not hasattr(shipped_coro, "OWA")
    assert shipped_coro.curves().data_extent_lod > shipped_coro.curves().iwa_lod


def test_data_extent_does_not_claim_a_dark_hole_boundary(shipped_coro):
    """Fails if the array edge again implies a usable low-contrast region.

    The shipped stellar map rises toward the array edge. The previous OWA
    sentinel xfailed because a geometric bound promised optical performance;
    the new API makes that distinction explicit and retains this physical
    counterexample. A factor of ten is deliberately below the measured rise.
    """
    extent = shipped_coro.curves().data_extent_lod
    reference = max(
        float(shipped_coro.curves().evaluate("raw_contrast", s))
        for s in np.linspace(shipped_coro.curves().iwa_lod + 1.0, 0.75 * extent, 8)
    )
    probes = np.linspace(0.75 * extent, extent, 9)
    values = np.asarray(shipped_coro.curves().evaluate("raw_contrast", probes))
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


def test_public_images_carry_header_grid(syn_coro_1d):
    """Fails if a public image drops geometry, reconstruction, measure or axes.

    Metadata are exact. The x=1 lod node is offset index 1/0.5=2.
    Its raw input pixels are the oracle; four machine epsilons of peak
    allow the padded FFT pair and weighted-normalization roundoff.
    """
    import equinox as eqx
    import jax.numpy as jnp
    from optixstuff.grids import Image

    from yippy._offax import OffAx

    coro = syn_coro_1d
    grid = coro.header.pixel_grid()
    numpy_reader = OffAx(
        coro.package.offaxis_psfs,
        coro.package.offsets_xy_lod,
        grid,
        True,
        True,
    )
    native_star = coro.stellar_map(None, 0.1, grid=None)
    native_source = coro.off_axis_psf(None, 1.0, None)
    batch_source = coro.source_psfs(None, jnp.array([1.0]), jnp.array([0.0])).image
    batch_star = coro.stellar_map(None, jnp.array([0.0, 0.1]))
    for product in (
        coro,
        coro._offaxis_backend,
        numpy_reader,
        native_star,
        coro.sky_trans,
    ):
        assert product.grid == grid
        assert not hasattr(product, "center_x")
        assert not hasattr(product, "center_y")
    rendered_source = coro.source_psfs(
        None, jnp.array([1.0]), jnp.array([0.0]), grid=grid
    ).image
    rendered_star = coro.stellar_map(None, 0.1, grid=grid)
    images = [
        numpy_reader.get_psf_by_offset_idx(2, 0),
        native_source,
        coro._offaxis_backend(1.0 * lod, 0.0 * lod),
        coro._offaxis_backend.get_psf_by_offset_idx(2, 0),
        batch_source,
        coro._offaxis_backend.create_psfs_parallel(jnp.array([1.0]), jnp.array([0.0])),
        native_star,
        batch_star,
        eqx.filter_jit(lambda x, y: coro.source_psfs(None, x, y))(
            jnp.array([1.0]), jnp.array([0.0])
        ).image,
        stellar_jit := eqx.filter_jit(lambda d: coro.stellar_map(None, d))(0.1),
    ]
    stellar = (native_star, batch_star, stellar_jit, rendered_star)
    for image in [*images, rendered_source, rendered_star]:
        assert isinstance(image, Image)
        assert image.grid == grid
        assert image.data.shape[-2:] == grid.shape
        assert len(image.leading_axes) == image.data.ndim - 2
        expected_measure = (
            Measure.density("grid") if image in images else Measure.integral()
        )
        assert (image.measure, image.quantity) == (
            expected_measure,
            "stellar_flux_fraction",
        )
        # Stellar halos declare linear (see test_shipped_stellar_map_stays_positive).
        expected = "linear" if any(image is s for s in stellar) else "band_limited"
        assert image.reconstruction == expected
    np.testing.assert_array_equal(
        coro.sky_trans.data,
        np.asarray(coro.package.sky_transmission, dtype=coro.sky_trans.data.dtype),
    )
    for image in (coro.sky_trans, coro.uniform_sky_map(None, grid=grid)):
        assert isinstance(image, Image) and image.grid == grid
        assert image.data.shape == grid.shape
        assert (image.measure, image.quantity) == (Measure.intensive(), "transmission")
        assert image.reconstruction == "linear"
        assert image.outside == "reflect"
    # Stellar halos continue past the field; off-axis PSFs vanish outside it.
    for image in (native_star, batch_star, rendered_star):
        assert image.outside == "reflect"
    for image in (native_source, batch_source, rendered_source):
        assert image.outside == "zero"
    # A coarser detector cannot promise to reconstruct the native bandwidth.
    coarse = PixelGrid.regular((32, 32), 0.5, "lod", frame=grid.frame)
    for image in (
        coro.source_psfs(None, jnp.array([1.0]), jnp.array([0.0]), grid=coarse).image,
        coro.stellar_map(None, 0.1, grid=coarse),
    ):
        assert image.reconstruction == "linear"
    actual = np.asarray(native_source.integrated().data)
    flat_idx = numpy_reader.offset_to_flat_idx[2, 0]
    expected = np.asarray(numpy_reader.flat_psfs[flat_idx], dtype=actual.dtype)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0,
        atol=4 * np.finfo(actual.dtype).eps * float(expected.max()),
    )


def test_detector_aperture_converges_to_encircled_flux(syn_coro_1d, synthetic_1d):
    """Fails if detector rendering silently uses linear PSF reconstruction.

    Solution verification: refine detector pitch by two at fixed angular extent.
    Pixel-box integration and fractional aperture coverage give second-order
    error against the analytic Gaussian encircled flux. Spike
    basis: the synthetic Gaussian's error/pitch**2 is below 0.16; allow 0.18.
    The observed-order band permits aperture-edge phase and reference quadrature
    error, while rejecting the nonvanishing linear-reconstruction bias.
    """
    import jax
    from optixstuff.apertures import Circular

    with jax.enable_x64():
        rule, separation = Circular(0.7), 2.0
        expected = synthetic_1d.expected_throughput(separation, rule.radius_lod)
        errors, sums = [], []
        pitches = (0.24, 0.12, 0.06)
        for pitch, size in zip(pitches, (32, 64, 128), strict=True):
            target = PixelGrid.regular(
                (size, size), pitch, "lod", frame=syn_coro_1d.grid.frame
            )
            psf = syn_coro_1d.off_axis_psf(None, separation, target)
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
    image = np.asarray(coro.stellar_map(None, 0.0, grid=None).data)

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
        psf = np.asarray(coro.off_axis_psf(None, separation, None).data)
        row, col = np.unravel_index(np.argmax(psf), psf.shape)
        a, b, c = np.log(psf[row, col - 1 : col + 2])
        peak_x = col + 0.5 * (a - c) / (a - 2 * b + c)
        assert abs(peak_x - (127.5 + separation / 0.25)) < 0.1


def test_public_geometric_leakage_change(shipped_coro):
    """Pin the change from coverage-weighted stellar pixels at four separations.

    The comparison uses the first stellar diameter and native YIP records
    interpolated at 2, 3, 5 and 10 lod. Allow 0.01 percentage point for
    rounding of the references and the active YIP storage precision.
    """
    from optixstuff.apertures import Circular, HalfMaximum

    from yippy._yip import _YIPData

    host = _YIPData(shipped_coro.package)
    for rule, native_leakage, change_percent in (
        (
            Circular(0.7),
            [1.62776811e-11, 2.10543705e-11, 3.61032578e-12, 3.63004298e-12],
            [-1.204, 1.137, -0.846, 7.988],
        ),
        (
            HalfMaximum(),
            [1.88133875e-11, 2.48969389e-11, 4.51811543e-12, 3.34974016e-12],
            [-1.610, 0.300, -0.030, 8.146],
        ),
    ):
        record = host._prepare_record(rule)
        leakage = record.evaluate("leakage_fraction", np.array([2.0, 3.0, 5.0, 10.0]))
        np.testing.assert_allclose(
            100 * (leakage / native_leakage - 1), change_percent, atol=0.01, rtol=0
        )


def test_shipped_stellar_map_stays_positive(shipped_coro):
    """Fails if the stellar map rings negative in the AAVC dark hole.

    A band-limited interpolant of the eight-decade stellar halo rang along the rows
    and columns of the bright pixels at the level of the hole floor (about 1e-12),
    and a consumer that clamps at zero then rendered exact-zero pixels and bands
    (2 to 11 percent of the hole at detector pitches of 0.45 lambda/D and finer).
    Every resampled value must be positive on a coarser and a finer grid.
    """
    frame = shipped_coro.stellar_grid.frame
    for n, pitch in ((144, 0.4532), (412, 0.1511)):
        grid = PixelGrid.regular((n, n), pitch, "lod", frame=frame)
        data = np.asarray(shipped_coro.stellar_map(None, 0.0, grid=grid).data)
        assert data.min() > 0.0


def test_shipped_off_axis_psfs_ring_only_at_the_noise_floor(shipped_coro):
    """Fails if band-limited resampling of an off-axis PSF rings more than measured.

    The off-axis PSFs stay band-limited. Near the dark-hole edge (2.5 to 4 lambda/D)
    on a grid finer than native the interpolant goes slightly negative; measured
    worst case is -7.5e-6 of the PSF peak (about 200 pixels at 3 lambda/D, pitch
    0.1511) and at most -1.1e-7 at pitch 0.4532. The bound is twice the measured size;
    this pins the size and does not claim the ringing is absent.
    """
    frame = shipped_coro.stellar_grid.frame
    xs = np.array([2.5, 3.0, 3.5, 4.0])
    ys = np.zeros_like(xs)
    for n, pitch, bound in ((144, 0.4532, 2e-6), (412, 0.1511, 1.5e-5)):
        grid = PixelGrid.regular((n, n), pitch, "lod", frame=frame)
        data = np.asarray(shipped_coro.source_psfs(None, xs, ys, grid=grid).image.data)
        worst = (data.min(axis=(-2, -1)) / data.max(axis=(-2, -1))).min()
        assert worst > -bound
