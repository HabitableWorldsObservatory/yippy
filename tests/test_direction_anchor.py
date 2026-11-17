"""Where a source at a known sky position lands on the array.

Every consumer that places a planet in a simulated frame inherits this mapping,
so an error here moves every planet in the stack by the same amount and no
downstream test that compares a frame against another frame can see it. The
assertions use a synthetic package whose PSF is a single Gaussian blob at a
known offset, which locates the peak with a quadratic fit to log intensity.
The 1e-5 pixel bound allows float32 FFT roundoff; a half-pixel shift fails it.

The expected column and row come from the package's declared centre and pixel
scale, ``column = center_x + x / pixscale`` and ``row = center_y + y / pixscale``,
not from any quantity the reader computes.
"""

import numpy as np
import pytest
from lod_unit import lod

# Offsets are whole pixel displacements from the half-integer optical axis.
GRID_ALIGNED_LOD = (1.0, 2.0, 3.0)


def _peak(coro, x_lod, y_lod):
    """The (row, column) of the brightest pixel for a source at a sky position."""
    psf = np.asarray(coro._offaxis_backend(x_lod * lod, y_lod * lod).data)
    row, col = np.unravel_index(np.argmax(psf), psf.shape)

    def vertex(values):
        a, b, c = np.log(values)
        return 0.5 * (a - c) / (a - 2 * b + c)

    return (
        row + vertex(psf[row - 1 : row + 2, col]),
        col + vertex(psf[row, col - 1 : col + 2]),
    )


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
        assert column == pytest.approx(expected_column, abs=1e-5, rel=0), (
            f"a source at x = +{separation} lambda/D landed on column {column}, "
            f"not {expected_column:.0f}"
        )
        assert row == pytest.approx(expected_row, abs=1e-5, rel=0), (
            f"a source on the x axis landed on row {row}, not {expected_row:.0f}; "
            "x and y are swapped"
        )


def test_a_source_at_positive_y_lands_on_the_expected_row(package, coro):
    """The y coordinate measures rows, and +y moves to a higher row index."""
    for separation in GRID_ALIGNED_LOD:
        row, column = _peak(coro, 0.0, separation)
        expected_column, expected_row = package.pixel_of(0.0, separation)
        assert row == pytest.approx(expected_row, abs=1e-5, rel=0), (
            f"a source at y = +{separation} lambda/D landed on row {row}, "
            f"not {expected_row:.0f}"
        )
        assert column == pytest.approx(expected_column, abs=1e-5, rel=0), (
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
    along_x = np.asarray(coro._offaxis_backend(2.0 * lod, 0.0 * lod).data)
    along_y = np.asarray(coro._offaxis_backend(0.0 * lod, 2.0 * lod).data)
    assert not np.allclose(along_x, along_y), (
        "a source displaced along x gives the same image as one displaced "
        "along y; the axes are transposed"
    )
    assert np.allclose(along_x, along_y.T, atol=along_x.max() * 1e-6), (
        "the two displacements should be transposes of one another for this "
        "symmetric synthetic package"
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
        assert midpoint == pytest.approx(package.center, abs=1e-5, rel=0), (
            f"+/-{separation} lambda/D land on columns {positive_column} and "
            f"{negative_column}, symmetric about {midpoint}, but the package "
            f"declares its axis at {package.center}"
        )


def test_negative_offsets_are_at_least_self_consistent_between_axes(package, coro):
    """Whatever the centre convention is, x and y must share it.

    A correction applied to only one axis would leave the response asymmetric.
    The tolerance is the 1e-5 pixel float32 peak-fit floor used above.
    """
    for separation in GRID_ALIGNED_LOD:
        _, negative_column = _peak(coro, -separation, 0.0)
        negative_row, _ = _peak(coro, 0.0, -separation)
        assert negative_column == pytest.approx(negative_row, abs=1e-5, rel=0), (
            f"a source at x = -{separation} lands on column {negative_column} "
            f"but one at y = -{separation} lands on row {negative_row}; the "
            "two axes disagree about the centre convention"
        )
