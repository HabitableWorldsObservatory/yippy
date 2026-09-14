"""Build a yield input package whose every quantity has a closed form.

The shipped packages are propagation products: correct answers for them are not
known independently of the code that reads them, so a test against one can only
pin what the code already does. This module writes a small package whose PSFs,
stellar maps and sky transmission are elementary functions, which makes the
throughput, core mean intensity, contrast and working angles computable by hand
inside a test.

The conventions encoded here are the ones the reader is expected to follow, so a
test that compares against them is checking the reader and not restating it:

* A package is a directory of five FITS files. ``offax_psf.fits`` holds one
  image per source position, and image ``i`` is what the instrument records for
  a source at ``offsets[i]``, so the blob sits away from the array center by
  the offset, not at it.
* Pixel and sky coordinates are related by ``column = center_x + x / pixscale``
  and ``row = center_y + y / pixscale``, with the center taken from the
  ``XCENTER`` and ``YCENTER`` header keywords. x measures columns and y
  measures rows.
* An off-axis PSF is a circular Gaussian of standard deviation ``psf_sigma_lod``
  whose total flux is ``psf_flux`` times an occulter transmission
  ``floor + (1 - floor) * (1 - exp(-(s / occulter_s0)^2))`` at source
  separation ``s``, so the flux inside a radius ``a`` centered on the blob is
  ``psf_flux * occulter(s) * (1 - exp(-a^2 / (2 sigma^2)))``. The occulter is
  what gives the package an inner working angle; without one the throughput
  curve is flat and the half-maximum crossing that defines the angle does not
  exist. The floor is small and positive so that the offset grid can include
  the optical axis.
* A stellar intensity map is constant at ``stellar_level`` per pixel, so any
  mean over any aperture is that constant.
* Sky transmission is ``1 - exp(-(r / sky_trans_r0)^2)`` in lambda/D.

Both a radially symmetric package (offsets along +x only) and a quarterly
symmetric one (offsets on an x-y grid) are available, because the reader takes
a different path through each and only a two-package test can catch an axis
swap between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import astropy.io.fits as pyfits
import numpy as np

# A package small enough to write and load inside a test, and coarse enough
# that a pixel-membership aperture is a visible error source rather than a
# rounding detail.
NPIX = 64
PIXSCALE = 0.25
CENTER = NPIX / 2
PSF_SIGMA_LOD = 0.5
PSF_FLUX = 1.0
STELLAR_LEVEL = 1e-10
SKY_TRANS_R0 = 2.0
OCCULTER_S0_LOD = 2.0
# A small residual transmission on axis. Without it the occulter transmits
# exactly nothing at zero separation, the tabulated throughput there is zero,
# and the reader's logarithmic contrast interpolant is non-finite, which fails
# construction inside scipy. A floor keeps the axis on the offset grid so that
# an on-axis query is a table lookup rather than an interpolation across the
# mirrored knots.
OCCULTER_FLOOR = 1e-6
DIAMS_LOD = (0.0, 0.1, 0.2)


@dataclass(frozen=True)
class SyntheticYIP:
    """A written package plus the constants a test needs to predict its answers."""

    path: Path
    npix: int
    pixscale: float
    center: float
    psf_sigma_lod: float
    psf_flux: float
    stellar_level: float
    sky_trans_r0: float
    occulter_s0_lod: float
    occulter_floor: float
    offsets: np.ndarray
    kind: str

    @property
    def max_offset_in_image(self) -> float:
        """Largest offset whose PSF center still lands inside the array."""
        return self.npix / 2 * self.pixscale

    def pixel_of(self, x_lod: float, y_lod: float) -> tuple[float, float]:
        """The (column, row) a source at ``(x_lod, y_lod)`` lands on."""
        return (
            self.center + x_lod / self.pixscale,
            self.center + y_lod / self.pixscale,
        )

    def occulter(self, separation_lod):
        """Closed-form occulter transmission at a source separation."""
        s = np.asarray(separation_lod, dtype=float)
        rise = 1.0 - np.exp(-((s / self.occulter_s0_lod) ** 2))
        return self.occulter_floor + (1.0 - self.occulter_floor) * rise

    def encircled_flux(self, radius_lod: float) -> float:
        """Closed-form flux fraction inside a radius centered on a PSF blob."""
        return self.psf_flux * (
            1.0 - np.exp(-(radius_lod**2) / (2.0 * self.psf_sigma_lod**2))
        )

    def expected_throughput(self, separation_lod, aperture_radius_lod=0.7):
        """Closed-form throughput: occulter transmission times encircled flux."""
        return self.occulter(separation_lod) * self.encircled_flux(aperture_radius_lod)

    def expected_iwa(self, max_offset_lod=None):
        """Closed-form inner working angle: the half-maximum throughput crossing.

        The reader defines the angle as the first separation at which the
        throughput curve reaches half its maximum over the tabulated offsets.
        The encircled-flux factor is common to both sides, so the crossing
        depends on the occulter alone.
        """
        s_max = self.offsets[:, 0].max() if max_offset_lod is None else max_offset_lod
        target = float(self.occulter(s_max)) / 2.0
        rise = (target - self.occulter_floor) / (1.0 - self.occulter_floor)
        return self.occulter_s0_lod * np.sqrt(-np.log(1.0 - rise))

    def sky_trans_at(self, radius_lod: float) -> float:
        """Closed-form sky transmission at a radius."""
        return 1.0 - np.exp(-((radius_lod / self.sky_trans_r0) ** 2))


def _grids(npix: int, pixscale: float, center: float):
    """Sky coordinates in lambda/D of every pixel center, as (x, y) arrays."""
    idx = np.arange(npix, dtype=float)
    x = (idx - center) * pixscale
    y = (idx - center) * pixscale
    return np.meshgrid(x, y, indexing="xy")


def _gaussian_psf(npix, pixscale, center, x0_lod, y0_lod, sigma_lod, flux):
    """A circular Gaussian of known total flux centered on a sky position.

    Normalized by the analytic integral rather than by its own sum, so that the
    array sum differs from ``flux`` only by the light that falls outside the
    array. A test can therefore use ``flux`` as truth for an aperture well
    inside the field, and can also measure the truncation.
    """
    xx, yy = _grids(npix, pixscale, center)
    r2 = (xx - x0_lod) ** 2 + (yy - y0_lod) ** 2
    amplitude = flux * pixscale**2 / (2.0 * np.pi * sigma_lod**2)
    return amplitude * np.exp(-r2 / (2.0 * sigma_lod**2))


def _header(npix, n_plane, pixscale, center, design):
    """A header carrying every keyword the reader recognizes."""
    h = pyfits.Header()
    h["DESIGN"] = (design, "synthetic analytic package")
    h["D"] = (6.0, "meters, telescope diameter")
    h["D_INSC"] = (5.0, "meters, inscribed diameter")
    h["PIXSCALE"] = (pixscale, "lambda/D per pixel")
    h["LAMBDA"] = (500.0, "nm, central wavelength")
    h["MINLAM"] = (450.0, "nm")
    h["MAXLAM"] = (550.0, "nm")
    h["XCENTER"] = (center, "pixels, column of the optical axis")
    h["YCENTER"] = (center, "pixels, row of the optical axis")
    h["OBSCURED"] = (0.0, "fraction of the pupil obscured")
    h["JITTER"] = (0.0, "mas")
    h["N_LAM"] = (1, "wavelengths")
    h["N_STAR"] = (n_plane, "stellar diameters")
    h["ZERNIKE"] = ("none", "aberration set")
    h["WFE"] = (0.0, "nm")
    return h


def write_synthetic_yip(
    directory: Path,
    kind: str = "1d",
    npix: int = NPIX,
    pixscale: float = PIXSCALE,
    psf_sigma_lod: float = PSF_SIGMA_LOD,
    psf_flux: float = PSF_FLUX,
    stellar_level: float = STELLAR_LEVEL,
    sky_trans_r0: float = SKY_TRANS_R0,
    occulter_s0_lod: float = OCCULTER_S0_LOD,
    occulter_floor: float = OCCULTER_FLOOR,
    min_offset_lod: float = 0.0,
    max_offset_lod: float = 6.0,
    offset_step_lod: float = 0.5,
) -> SyntheticYIP:
    """Write a five-file package and return the constants that describe it.

    Args:
        directory: where to write; created if absent.
        kind: ``"1d"`` for offsets along +x only (radially symmetric), or
            ``"2dq"`` for an x-y grid of non-negative offsets (quarterly
            symmetric). The reader takes a different path through each.
        npix: image side in pixels.
        pixscale: lambda/D per pixel.
        psf_sigma_lod: off-axis PSF standard deviation in lambda/D.
        psf_flux: total flux of every off-axis PSF.
        stellar_level: the constant value of every stellar intensity map.
        sky_trans_r0: scale of the sky transmission function in lambda/D.
        occulter_s0_lod: scale of the occulter transmission in lambda/D, which
            is what gives the package an inner working angle.
        occulter_floor: residual on-axis transmission, which must be positive
            so that a tabulated offset at zero separation does not give a
            throughput of exactly zero.
        min_offset_lod: smallest source offset to tabulate.
        max_offset_lod: largest source offset to tabulate.
        offset_step_lod: spacing of the offset grid.

    Returns:
        A :class:`SyntheticYIP` describing what was written.

    Raises:
        ValueError: if ``kind`` is unknown or ``min_offset_lod`` is not
            positive.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    center = npix / 2
    if occulter_floor <= 0:
        raise ValueError(
            "occulter_floor must be positive; a zero-separation offset would "
            "otherwise give zero throughput and a non-finite log-contrast "
            "interpolant, which fails construction inside scipy"
        )

    stop = max_offset_lod + 0.5 * offset_step_lod
    steps = np.arange(min_offset_lod, stop, offset_step_lod)
    if kind == "1d":
        offsets = np.column_stack([steps, np.zeros_like(steps)])
        offset_payload = steps.astype(np.float64)
    elif kind == "2dq":
        # The grid includes both axes, so a source anywhere on either axis
        # lands on a knot. A grid that started away from an axis would make
        # every on-axis query an interpolation across the reader's mirrored
        # knots, which looks like a reader error and is not one.
        xx, yy = np.meshgrid(steps, steps, indexing="ij")
        offsets = np.column_stack([xx.ravel(), yy.ravel()])
        offset_payload = offsets.astype(np.float64)
    else:
        raise ValueError(f"kind must be '1d' or '2dq', got {kind!r}")

    separations = np.hypot(offsets[:, 0], offsets[:, 1])
    rise = 1.0 - np.exp(-((separations / occulter_s0_lod) ** 2))
    occulter = occulter_floor + (1.0 - occulter_floor) * rise
    psfs = np.stack(
        [
            _gaussian_psf(npix, pixscale, center, x, y, psf_sigma_lod, psf_flux * t_occ)
            for (x, y), t_occ in zip(offsets, occulter, strict=True)
        ]
    ).astype(np.float64)
    pyfits.PrimaryHDU(
        psfs, header=_header(npix, len(psfs), pixscale, center, "synthetic-offax")
    ).writeto(directory / "offax_psf.fits", overwrite=True)
    pyfits.PrimaryHDU(offset_payload).writeto(
        directory / "offax_psf_offset_list.fits", overwrite=True
    )

    # One constant map per stellar diameter: the mean over any aperture is the
    # constant, so core mean intensity has an exact expected value.
    diams = np.array(DIAMS_LOD, dtype=np.float64)
    stellar = np.full((len(diams), npix, npix), stellar_level, dtype=np.float64)
    pyfits.PrimaryHDU(
        stellar,
        header=_header(npix, len(diams), pixscale, center, "synthetic-stellar"),
    ).writeto(directory / "stellar_intens.fits", overwrite=True)
    pyfits.PrimaryHDU(diams).writeto(
        directory / "stellar_intens_diam_list.fits", overwrite=True
    )

    xx, yy = _grids(npix, pixscale, center)
    radius = np.sqrt(xx**2 + yy**2)
    sky = 1.0 - np.exp(-((radius / sky_trans_r0) ** 2))
    pyfits.PrimaryHDU(
        sky.astype(np.float64),
        header=_header(npix, 1, pixscale, center, "synthetic-sky"),
    ).writeto(directory / "sky_trans.fits", overwrite=True)

    return SyntheticYIP(
        path=directory,
        npix=npix,
        pixscale=pixscale,
        center=center,
        psf_sigma_lod=psf_sigma_lod,
        psf_flux=psf_flux,
        stellar_level=stellar_level,
        sky_trans_r0=sky_trans_r0,
        occulter_s0_lod=occulter_s0_lod,
        occulter_floor=occulter_floor,
        offsets=offsets,
        kind=kind,
    )
