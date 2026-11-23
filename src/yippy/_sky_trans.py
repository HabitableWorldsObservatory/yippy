"""This module handles the sky transmission map."""

import numpy as np
from optixstuff.grids import Image, Measure


def sky_transmission_image(package) -> Image:
    """Linearly reconstructed dimensionless extended-source transmission.

    Args:
        package: Coronagraph package whose ``sky_transmission`` array and header
            grid are wrapped.

    Returns:
        Dimensionless transmission image on the sky transmission pixel grid,
        reconstructed linearly between pixels.
    """
    return Image(
        np.asarray(package.sky_transmission, dtype=float),
        package.header.sky_transmission.pixel_grid(),
        (),
        Measure.intensive(),
        "transmission",
        reconstruction="linear",
    )
