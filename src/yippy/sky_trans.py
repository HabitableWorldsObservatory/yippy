"""This module handles the sky transmission map."""

import numpy as np
from hwoutils.grids import Image, Measure


def sky_transmission_image(package) -> Image:
    """Linearly reconstructed dimensionless extended-source transmission."""
    return Image(
        np.asarray(package.sky_transmission, dtype=float),
        package.header.sky_transmission.pixel_grid(),
        (),
        Measure.intensive(),
        "transmission",
        reconstruction="linear",
        outside="reflect",
    )
