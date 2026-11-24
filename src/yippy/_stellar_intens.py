"""Module for handling stellar intensity data from stellar_intens files."""

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from lod_unit import lod
from optixstuff.grids import Frame, Image
from scipy.interpolate import CubicSpline, PchipInterpolator

from ._precision import float_dtype
from .sampling import STELLAR_RECONSTRUCTION, fraction_image, sample_units

STELLAR_INTERPOLATION_REVISION = 1


def stellar_knot_index(axis, diameter, xp):
    """Index the preceding knot, using the end knots outside the fitted axis.

    Args:
        axis: Ascending knot diameters.
        diameter: Query diameter(s).
        xp: Array namespace (NumPy or ``jax.numpy``).

    Returns:
        Index of the knot at or below each query, clamped to the axis.
    """
    return xp.clip(xp.searchsorted(axis, diameter, side="right") - 1, 0, len(axis) - 1)


def bounded_stellar_pchip(spline, diameter, samples, knot, xp):
    """Clamp the query and constrain the result to its bracketing samples.

    Args:
        spline: Fitted spline over the knot axis (``spline.x`` is the axis).
        diameter: Query diameter(s).
        samples: Measured samples along the knot axis.
        knot: Index of the preceding knot for each query.
        xp: Array namespace (NumPy or ``jax.numpy``).

    Returns:
        The interpolated intensity, never outside its two bracketing samples.
    """
    axis = spline.x
    left = xp.minimum(knot, len(axis) - 2)
    first, second = samples[left], samples[left + 1]
    value = spline(xp.clip(diameter, axis[0], axis[-1]))
    low, high = xp.minimum(first, second), xp.maximum(first, second)
    # Select instead of clip so a value sitting on a bound keeps unit slope.
    return xp.where(value < low, low, xp.where(value > high, high, value))


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

        kind = package.header.stellar_intensity.sample_kind
        self.measure, factor = sample_units(kind, self.grid)
        self.spline_samples = (
            self.psfs
            if kind != "mean"
            else np.multiply(self.psfs, factor, dtype=np.float64)
        )
        dtype = float_dtype()
        tiny = np.finfo(dtype).tiny
        subnormal = (self.psfs > 0) & (
            (self.psfs < tiny) | (self.spline_samples < tiny)
        )
        count = np.count_nonzero(subnormal)
        if count:
            path = package._files["stellar_intensity"]
            raise ValueError(
                f"{path}: {count} positive stellar sample(s) below the smallest "
                f"normal number for the working dtype {dtype}"
            )

        self.zero_pixels = np.any(self.psfs == 0, axis=0)
        self.zero_pixels.setflags(write=False)
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
        axis = self.ln_interp.x
        diameter = np.asarray(stellar_diam.value, dtype=axis.dtype)
        knot = stellar_knot_index(axis, diameter, np)
        at_knot = (diameter == axis[knot])[..., None, None]
        data = np.exp(self.ln_interp(diameter))
        if self.linear_interp is not None:
            linear = bounded_stellar_pchip(
                self.linear_interp, diameter, self.spline_samples, knot, np
            )
            data = np.where(self.zero_pixels, linear, data)
            outside = (diameter < axis[0]) | (diameter > axis[-1])
            at_knot = at_knot | (outside[..., None, None] & self.zero_pixels)
        return np.where(at_knot, self.spline_samples[knot], data)
