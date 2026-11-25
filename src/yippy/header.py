"""Module for handling FITS header data."""

import re
from dataclasses import dataclass

import astropy.units as u
from astropy.io import fits
from lod_unit import lod
from optixstuff.grids import Frame, PixelGrid

from .logger import logger


@dataclass(frozen=True)
class HeaderData:
    """FITS package metadata with units attached.

    xcenter and ycenter retain the header's 00LL pixel corner counts.
    Use pixel_grid() for pixel-center coordinates and the reference frame.
    D_INSC remains metadata; it does not change the native circumscribed frame.

    In offax_psf.fits, MINLAM below MAXLAM declares the flat spectral average
    over those limits on the lambda/D grid at LAMBDA. Place the grid at each
    observing band's centre, as described by Stark and Krist (2019),
    "Standardized Coronagraph Parameters for Input into Yield Calculations".
    A producer computing one wavelength must write equal limits. Absent limits
    also indicate one wavelength. stellar_intens.fits must declare matching
    edges, while sky_trans.fits may differ.

    Attributes:
        simple: FITS SIMPLE flag.
        bitpix: FITS BITPIX data type code.
        naxis: Number of FITS axes.
        naxis1: Length of the first (x) axis, in pixels.
        naxis2: Length of the second (y) axis, in pixels.
        naxis3: Length of the third axis, zero when absent.
        design: Optical design name from DESIGN.
        diameter: Circumscribed telescope diameter from D; a length (default unit m).
        diameter_inscribed: Inscribed telescope diameter from D_INSC; a length (default
            unit m).
        pixscale: Pixel scale from PIXSCALE, in lambda/D per pixel.
        lambda0: Reference wavelength from LAMBDA (default unit um).
        minlam: Shortest bandpass wavelength from MINLAM (default unit um).
        maxlam: Longest bandpass wavelength from MAXLAM (default unit um).
        xcenter: Optical axis in x, in pixel corner counts (XCENTER).
        ycenter: Optical axis in y, in pixel corner counts (YCENTER).
        obscured: Obscured aperture-area fraction from OBSCURED.
        jitter: RMS jitter per axis from JITTER (default unit mas).
        n_lam: Number of wavelength samples from N_LAM, an optional consistency
            check outside the specification.
        n_star: Number of stellar samples from N_STAR.
        zernike: Aberration set name from ZERNIKE.
        wfe: Wavefront error from WFE (default unit pm).
        sample_kind: Intensity sampling convention, ``"point"``, ``"integral"`` or
            ``"mean"`` (SAMPKIND).
        diameter_convention: Diameter convention of the native frame.
        coordinate_model: Coordinate model of the native frame.
    """

    simple: bool
    bitpix: int
    naxis: int
    naxis1: int
    naxis2: int
    naxis3: int
    design: str
    diameter: u.Quantity | None = None
    diameter_inscribed: u.Quantity | None = None
    pixscale: u.Quantity | None = None
    lambda0: u.Quantity | None = None
    minlam: u.Quantity | None = None
    maxlam: u.Quantity | None = None
    xcenter: float | None = None
    ycenter: float | None = None
    obscured: float | None = None
    jitter: u.Quantity | None = None
    n_lam: int | None = None
    n_star: int | None = None
    zernike: str | None = None
    wfe: u.Quantity | None = None
    sample_kind: str = "point"
    diameter_convention: str = "circumscribed"
    coordinate_model: str = "paraxial_entrance_tilt"

    def __post_init__(self):
        """Protect the underlying quantity arrays as well as the header fields."""
        if self.sample_kind not in ("point", "integral", "mean"):
            raise ValueError("SAMPKIND must be point, integral or mean")
        for value in vars(self).values():
            if isinstance(value, u.Quantity):
                value.flags.writeable = False

    def pixel_grid(self, shape=None) -> PixelGrid:
        """Native grid; FITS 00LL centers count from the outer pixel corner.

        Missing centers use the geometric midpoint of the supplied (ny, nx)
        shape, or the header image shape. Native FITS products use circumscribed D.

        Args:
            shape: Optional (ny, nx) shape of the grid in pixels; None uses the header
                image shape.

        Returns:
            The native pixel grid in lambda/D with its reference frame.
        """
        shape = (self.naxis2, self.naxis1) if shape is None else tuple(shape)
        for key, value in (
            ("PIXSCALE", self.pixscale),
            ("LAMBDA", self.lambda0),
            ("D", self.diameter),
        ):
            if value is None:
                raise ValueError(f"{key} is required to build a pixel grid")
        step = self.pixscale.to_value(lod / u.pix)
        return PixelGrid.from_corner_count(
            shape,
            shape[1] / 2 if self.xcenter is None else self.xcenter,
            shape[0] / 2 if self.ycenter is None else self.ycenter,
            basis=((step, 0.0), (0.0, step)),
            unit="lod",
            frame=Frame(
                self.lambda0.to_value(u.nm),
                self.diameter.to_value(u.m),
                self.diameter_convention,
                self.coordinate_model,
            ),
        )

    @staticmethod
    def extract_unit(comment: str, default_unit: u.Unit, key: str) -> u.Unit:
        """Extract a unit from the comment string of a FITS header entry.

        This attempts to read the FITS header comment and search for an astropy
        Unit. This seems fairly robust, but it may not catch all cases in which
        case a warning is raised. It is also possible that the unit is not
        explicitly stated in the comment, in which case a default unit is
        returned.

        Args:
            comment: The comment string associated with a FITS header key.
            default_unit: The default unit to return if no unit is found in the
                comment.
            key: The key of the header entry to extract the unit from.

        Returns:
            The extracted astropy unit or the default unit if no recognizable
            unit is found.

        Raises:
            Warning:
                Logs a warning if no unit could be extracted and the default
                unit is used.
        """
        # Define a regex pattern for commonly expected units
        unit_patterns = {
            "nm": u.nm,
            "micron": u.micron,
            "microns": u.micron,
            "um": u.um,
            "mm": u.mm,
            "cm": u.cm,
            "meter": u.m,
            "m": u.m,
            "arcsec": u.arcsec,
            "arcmin": u.arcmin,
            "deg": u.deg,
            "mas": u.mas,
            "pm": u.pm,
        }

        # Match for patterns that look like compound units, e.g., "mas/pixel"
        compound_unit_pattern = r"\b(\w+)/(\w+)\b"
        match = re.search(compound_unit_pattern, comment, re.IGNORECASE)
        if match:
            num_unit, den_unit = match.groups()
            num_astropy_unit = unit_patterns.get(num_unit.lower(), None)
            den_astropy_unit = unit_patterns.get(den_unit.lower(), None)
            if num_astropy_unit and den_astropy_unit:
                logger.debug(
                    f"Extracted compound unit {num_astropy_unit}/{den_astropy_unit}"
                    f' for {key} from "{comment}"'
                )
                return num_astropy_unit / den_astropy_unit

        # Search for any of these single units in the comment
        for pattern, unit in unit_patterns.items():
            if re.search(r"\b" + re.escape(pattern) + r"\b", comment, re.IGNORECASE):
                logger.debug(f'Extracted {unit} from "{comment}" for {key}')
                return unit

        logger.warning(
            f"Using default unit for {key}: {default_unit}. "
            f'Could not extract unit from comment: "{comment}"'
        )
        return default_unit

    @staticmethod
    def get_header_value(header: fits.Header, key: str, default_unit: u.Unit):
        """Retrieves a header value by key and converts it to a specified unit.

        Args:
            header: The FITS header from which to retrieve the value.
            key: The key of the header entry to retrieve.
            default_unit: The unit to associate with the returned value when the
                entry comment names none.

        Returns:
            The header value with the unit from its comment (or the default
            unit), or None if the key is not present.
        """
        if key in header:
            value = float(header[key])
            unit = HeaderData.extract_unit(header.comments[key], default_unit, key)
            return value * unit
        else:
            return None

    @staticmethod
    def from_fits_header(header: fits.Header):
        """Parses a FITS header.

        This parses the FITS header to initialize the HeaderData class,
        checking for any unhandled fields.

        Args:
            header: The FITS header to parse.

        Returns:
            An initialized HeaderData object populated with values from the FITS
            header.

        Raises:
            Warning:
                Logs a warning for any unhandled fields in the FITS header.
        """
        recognized_keys = set(
            [
                "SIMPLE",
                "BITPIX",
                "NAXIS",
                "NAXIS1",
                "NAXIS2",
                "NAXIS3",
                "DESIGN",
                "D",
                "D_INSC",
                "PIXSCALE",
                "SAMPKIND",
                "LAMBDA",
                "MINLAM",
                "MAXLAM",
                "XCENTER",
                "YCENTER",
                "OBSCURED",
                "JITTER",
                "N_LAM",
                "N_STAR",
                "ZERNIKE",
                "WFE",
            ]
        )
        header_keys = set(header.keys())

        unhandled_keys = header_keys - recognized_keys
        if unhandled_keys:
            logger.warning(f"Unhandled header fields: {unhandled_keys}")

        return HeaderData(
            simple=header.get("SIMPLE", False) in (True, "T"),
            bitpix=int(header.get("BITPIX", 0)),
            naxis=int(header.get("NAXIS", 0)),
            naxis1=int(header.get("NAXIS1", 0)),
            naxis2=int(header.get("NAXIS2", 0)),
            naxis3=int(header.get("NAXIS3", 0)),
            design=header.get("DESIGN", ""),
            diameter=HeaderData.get_header_value(header, "D", u.meter),
            diameter_inscribed=HeaderData.get_header_value(header, "D_INSC", u.meter),
            pixscale=(
                None if "PIXSCALE" not in header else header["PIXSCALE"] * lod / u.pix
            ),
            lambda0=HeaderData.get_header_value(header, "LAMBDA", u.micron),
            minlam=HeaderData.get_header_value(header, "MINLAM", u.micron),
            maxlam=HeaderData.get_header_value(header, "MAXLAM", u.micron),
            sample_kind=header.get("SAMPKIND", "point"),
            xcenter=header.get("XCENTER", None),
            ycenter=header.get("YCENTER", None),
            obscured=header.get("OBSCURED", None),
            jitter=HeaderData.get_header_value(header, "JITTER", u.mas),
            n_lam=header.get("N_LAM", None),
            n_star=header.get("N_STAR", None),
            zernike=header.get("ZERNIKE", ""),
            wfe=HeaderData.get_header_value(header, "WFE", u.pm),
        )


def lod_inscribed_to_circumscribed(separation_lod, frame: Frame, diameter_inscribed_m):
    """Convert inscribed lambda/D offsets to the supplied circumscribed frame.

    Pass header.diameter_inscribed.to_value("m") for diameter_inscribed_m.
    The separation may be an array or a traced scalar.

    Args:
        separation_lod: Offsets in lambda/D of the inscribed diameter; an array or a
            traced scalar.
        frame: Circumscribed frame supplying the telescope diameter in m.
        diameter_inscribed_m: Inscribed telescope diameter in m; positive and no larger
            than the frame diameter.

    Returns:
        The offsets in lambda/D of the circumscribed frame, with the shape of
        ``separation_lod``.
    """
    if frame.diameter_convention != "circumscribed":
        raise ValueError("frame must use the circumscribed diameter")
    if not 0 < diameter_inscribed_m <= frame.diameter_m:
        raise ValueError("inscribed diameter must be positive and no larger than D")
    return separation_lod * (frame.diameter_m / diameter_inscribed_m)
