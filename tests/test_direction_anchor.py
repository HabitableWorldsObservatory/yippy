"""Where a source at a known sky position lands on the array.

Every consumer that places a planet in a simulated frame inherits this mapping,
so an error here moves every planet in the stack by the same amount and no
downstream test that compares a frame against another frame can see it. The
assertions use a synthetic package whose PSF is a single Gaussian blob at a
known offset, which makes the expected pixel an exact integer at grid-aligned
offsets and leaves only a sign error or an axis swap able to fail them.

The expected column and row come from the package's declared centre and pixel
scale, ``column = center_x + x / pixscale`` and ``row = center_y + y / pixscale``,
not from any quantity the reader computes.
"""

import numpy as np
import pytest
from lod_unit import lod

# Offsets that are an exact whole number of pixels at the synthetic pixel
# scale, so argmax has a single unambiguous answer.
GRID_ALIGNED_LOD = (1.0, 2.0, 3.0)


def _peak(coro, x_lod, y_lod):
    """The (row, column) of the brightest pixel for a source at a sky position."""
    psf = np.asarray(coro.offax(x_lod * lod, y_lod * lod))
    return np.unravel_index(np.argmax(psf), psf.shape)


@pytest.fixture(params=["1d", "2d"])
def kind(request):
    """Both package symmetries take different paths through the reader."""
    return request.param


@pytest.fixture
def package(kind, synthetic_1d, synthetic_2d):
    """The synthetic package of the requested symmetry."""
    return synthetic_1d if kind == "1d" else synthetic_2d


@pytest.fixture
def coro(kind, syn_coro_1d, syn_coro_2d):
    """The coronagraph built over the requested package."""
    return syn_coro_1d if kind == "1d" else syn_coro_2d


def test_a_source_at_positive_x_lands_on_the_expected_column(package, coro):
    """The x coordinate measures columns, and +x moves to a higher column index.

    Only a phasor sign error or a row and column swap can fail this, which is
    the point. Both are mistakes that leave every other property of the image
    intact, so no test of flux, shape or smoothness can see them.
    """
    for separation in GRID_ALIGNED_LOD:
        row, column = _peak(coro, separation, 0.0)
        expected_column, expected_row = package.pixel_of(separation, 0.0)
        assert column == expected_column, (
            f"a source at x = +{separation} lambda/D landed on column {column}, "
            f"not {expected_column:.0f}"
        )
        assert row == expected_row, (
            f"a source on the x axis landed on row {row}, not {expected_row:.0f}; "
            "x and y are swapped"
        )


def test_a_source_at_positive_y_lands_on_the_expected_row(package, coro):
    """The y coordinate measures rows, and +y moves to a higher row index."""
    for separation in GRID_ALIGNED_LOD:
        row, column = _peak(coro, 0.0, separation)
        expected_column, expected_row = package.pixel_of(0.0, separation)
        assert row == expected_row, (
            f"a source at y = +{separation} lambda/D landed on row {row}, "
            f"not {expected_row:.0f}"
        )
        assert column == expected_column, (
            f"a source on the y axis landed on column {column}, "
            f"not {expected_column:.0f}; x and y are swapped"
        )


def test_the_two_axes_are_not_transposed(package, coro):
    """Displacing in x and displacing in y must not give the same image.

    A reader that swapped its axes would still pass a test that only ever
    moves a source along one of them, and would still conserve flux, and would
    still interpolate smoothly. This is the cheapest assertion that separates
    the two.
    """
    along_x = np.asarray(coro.offax(2.0 * lod, 0.0 * lod))
    along_y = np.asarray(coro.offax(0.0 * lod, 2.0 * lod))
    assert not np.allclose(along_x, along_y), (
        "a source displaced along x gives the same image as one displaced "
        "along y; the axes are transposed"
    )
    assert np.allclose(along_x, along_y.T, atol=along_x.max() * 1e-6), (
        "the two displacements should be transposes of one another for this "
        "symmetric synthetic package"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The response is symmetric about (npix - 1) / 2 because negative "
        "offsets are produced by a whole-array flip, while the coordinate "
        "conversion places the optical axis at npix / 2, which is what the "
        "package header declares. The two differ by half a pixel each way, so "
        "a source at -s lands one pixel closer to the axis than a source at "
        "+s. Reproduced on the shipped eac1_aavc_2d package, whose header "
        "gives XCENTER = 128.0 while its response is symmetric about 127.5. "
        "Fixing it means choosing one convention and sweeping every consumer "
        "that derives a planet position from these PSFs, so the contract is "
        "written here and the choice is escalated rather than made locally."
    ),
)
def test_the_response_is_symmetric_about_the_declared_centre(package, coro):
    """A source at -s must land as far from the axis as one at +s.

    The package declares where its optical axis is, in the header and on the
    loaded object. Whatever that declaration says, the response either agrees
    with it or the declaration is wrong, and a consumer converting a planet's
    sky position into a pixel has no way to tell which.
    """
    for separation in GRID_ALIGNED_LOD:
        _, positive_column = _peak(coro, separation, 0.0)
        _, negative_column = _peak(coro, -separation, 0.0)
        midpoint = (positive_column + negative_column) / 2.0
        assert midpoint == pytest.approx(package.center), (
            f"+/-{separation} lambda/D land on columns {positive_column} and "
            f"{negative_column}, symmetric about {midpoint}, but the package "
            f"declares its axis at {package.center}"
        )


def test_negative_offsets_are_at_least_self_consistent_between_axes(package, coro):
    """Whatever the centre convention is, x and y must share it.

    This holds today and is worth keeping even while the convention above is
    unresolved: a fix that corrected one axis and not the other would leave the
    response asymmetric in a way that is far harder to notice than the present
    symmetric-but-offset behavior.
    """
    for separation in GRID_ALIGNED_LOD:
        _, negative_column = _peak(coro, -separation, 0.0)
        negative_row, _ = _peak(coro, 0.0, -separation)
        assert negative_column == negative_row, (
            f"a source at x = -{separation} lands on column {negative_column} "
            f"but one at y = -{separation} lands on row {negative_row}; the "
            "two axes disagree about the centre convention"
        )
