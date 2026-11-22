"""Module for handling stellar intensity data from stellar_intens files."""

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from lod_unit import lod
from optixstuff.grids import Frame, Image
from scipy.interpolate import CubicSpline

from .sampling import STELLAR_RECONSTRUCTION, fraction_image, sample_units


class StellarIntens:
    """Diameter interpolator fitted to the native stellar intensity product."""

    def __init__(self, package) -> None:
        """Build the host spline without modifying the measured values."""
        psfs = package.stellar_intensity
        self.grid = package.header.stellar_intensity.pixel_grid()
        self.diams = package.stellar_diameters_lod * lod

        # Store a copy of the original PSFs
        kind = package.header.stellar_intensity.sample_kind
        self.measure, factor = sample_units(kind, self.grid)
        self.psfs = psfs.copy()

        # Interpolate stellar data in logarithmic space to ensure non-negative
        # interpolated values. The abscissa must be strictly increasing and the
        # ordinate positive, so a zero diameter and any zero intensity are
        # nudged to the smallest representable positive value for the spline
        # only. Substituting into self.diams instead would make a caller
        # reading the list back see a diameter the package does not have.
        interp_diams = self.diams.value.copy()
        interp_diams[interp_diams == 0] = np.finfo(np.float32).eps
        self.psfs[self.psfs == 0] = np.finfo(np.float32).eps
        if kind == "mean":
            self.psfs *= factor
        self.ln_interp = CubicSpline(interp_diams, np.log(self.psfs))

    def __call__(self, stellar_diam: Quantity, lam=None, D=None) -> Image:
        """Returns the stellar intensity map at a specified stellar diameter.

        Stellar intensity is interpolated based on a specified diameter using the
        previously configured cubic spline interpolator. The interpolation considers
        the logarithm of the intensity to ensure that all computed values are positive.

        Args:
            stellar_diam (Quantity):
                The desired stellar diameter for which to retrieve the intensity map, in
                units of lambda/D.
            lam (Quantity, optional):
                Wavelength of observation, required if stellar_diam is in angular units.
            D (Quantity, optional):
                Diameter of the telescope, required if stellar_diam is in angular units.

        Returns:
            Image:
                Native fraction of unocculted stellar flux, with diameter
                axes preceding (y, x).
        """
        data = self._evaluate_array(stellar_diam, lam, D)
        axes = tuple(f"diameter_{i}" for i in range(data.ndim - 2))
        return fraction_image(
            data,
            self.grid,
            axes,
            self.measure,
            reconstruction=STELLAR_RECONSTRUCTION,
        )

    def _evaluate_array(self, stellar_diam, lam=None, D=None):
        """Host spline values, retaining FITS precision for performance tables."""
        if stellar_diam.unit != lod:
            frame = Frame(lam.to_value(u.nm), D.to_value(u.m), "circumscribed")
            stellar_diam = (
                stellar_diam.to_value(u.arcsec)
                / frame.lod_to_arcsec(lam.to_value(u.nm))
                * lod
            )
        return np.exp(self.ln_interp(stellar_diam))
