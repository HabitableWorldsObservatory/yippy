"""Module for handling stellar intensity data from stellar_intens files."""

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from lod_unit import lod
from optixstuff.grids import Frame, Image
from scipy.interpolate import CubicSpline, PchipInterpolator

from .sampling import STELLAR_RECONSTRUCTION, fraction_image, sample_units


class StellarIntens:
    """Diameter interpolator fitted to the native stellar intensity product."""

    def __init__(self, package) -> None:
        """Build the host spline without modifying the measured values.

        Args:
            package: Coronagraph package supplying the stellar intensity maps,
                their stellar diameters (lambda/D) and the pixel grid.
        """
        self.psfs = package.stellar_intensity
        if np.any(self.psfs < 0) or not np.all(np.isfinite(self.psfs)):
            path = package._files["stellar_intensity"]
            raise ValueError(
                f"{path}: stellar intensity samples must be finite and nonnegative"
            )
        self.grid = package.header.stellar_intensity.pixel_grid()
        self.diams = package.stellar_diameters_lod * lod

        self.zero_pixels = np.any(self.psfs == 0, axis=0)
        self.zero_pixels.setflags(write=False)
        kind = package.header.stellar_intensity.sample_kind
        self.measure, factor = sample_units(kind, self.grid)
        self.spline_samples = self.psfs if kind != "mean" else self.psfs * factor
        diams_lod = self.diams.value
        self.linear_interp = None
        log_samples = self.spline_samples
        if self.zero_pixels.any():
            self.linear_interp = PchipInterpolator(diams_lod, self.spline_samples)
            log_samples = np.where(self.zero_pixels, 1, log_samples)
        self.ln_interp = CubicSpline(diams_lod, np.log(log_samples))

    def __call__(self, stellar_diam: Quantity, lam=None, D=None) -> Image:
        """Returns the stellar intensity map at a specified stellar diameter.

        Pixels with positive samples at every diameter use a cubic spline in log
        intensity. Pixels with any zero sample use PCHIP in linear intensity.

        Args:
            stellar_diam:
                The desired stellar diameter for which to retrieve the intensity map, in
                units of lambda/D.
            lam:
                Wavelength of observation, required if stellar_diam is in angular units.
            D:
                Diameter of the telescope, required if stellar_diam is in angular units.

        Returns:
            Image:
                Native fraction of unocculted stellar flux, with diameter
                axes preceding (y, x). The values are point samples declaring
                the cubic spline; read them onto a detector with
                ``yippy.sampling.stellar_on_grid``, which applies the
                positivity floor that ``optixstuff.coronagraph.on_grid``
                alone does not.
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
        data = np.exp(self.ln_interp(stellar_diam))
        if self.linear_interp is not None:
            first, last = self.linear_interp.x[[0, -1]]
            diameter = np.clip(stellar_diam.value, first, last)
            linear = self.linear_interp(diameter)
            # Use the measured endpoints to avoid roundoff in the polynomial.
            linear = np.where(
                (diameter == first)[..., None, None], self.spline_samples[0], linear
            )
            linear = np.where(
                (diameter == last)[..., None, None], self.spline_samples[-1], linear
            )
            data = np.where(self.zero_pixels, linear, data)
        return data
