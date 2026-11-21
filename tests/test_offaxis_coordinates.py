"""Host coordinate inputs use the authoritative off-axis grid and frame."""

import astropy.units as u
import numpy as np
from lod_unit import lod
from optixstuff.grids import Frame, PixelGrid

from yippy._offax import OffAx


def test_offaxis_coordinate_inputs_use_grid_and_frame():
    """Exact affine algebra and 8-epsilon angular roundoff; no PSF interpolation."""

    class Coordinates(OffAx):
        def __init__(self):
            self.grid = PixelGrid(
                (8, 10),
                3.0,
                2.0,
                ((0.5, 0.25), (-0.25, 0.5)),
                "lod",
                Frame(600.0, 6.0, "circumscribed"),
            )

        def create_psf(self, x, y):
            return np.array([x, y])

        def create_psfs(self, x, y):
            return np.array([x, y])

    off = Coordinates()
    np.testing.assert_array_equal(off(5 * u.pix, 6 * u.pix), [2.0, 1.5])
    np.testing.assert_array_equal(
        off(np.array([3, 5]) * u.pix, np.array([2, 6]) * u.pix), [[0, 2], [0, 1.5]]
    )
    np.testing.assert_array_equal(off(2 * lod, -1 * lod), [2, -1])
    from hwoutils.constants import nm2m, rad2arcsec

    arcsec_per_lod = 600 * nm2m / 6 * rad2arcsec
    np.testing.assert_allclose(
        off(2 * arcsec_per_lod * u.arcsec, -arcsec_per_lod * u.arcsec),
        [2, -1],
        rtol=8 * np.finfo(float).eps,
    )
