"""Serialize caller-supplied synthetic coronagraph data for tests."""

from .writer import write_yip


def write_synthetic_yip(path, *, grid, offsets_xy_lod, psfs, stellar, sky, headers):
    """Write fixture arrays; stellar is a pair of (diameters_lod, intensity_maps).

    The grid describes the native image coordinates. Headers can supply a
    common FITS Header or per-product cards, including distinct product grids.
    The caller supplies all radiometry, source coordinates and diameters.

    Args:
        path: Output file path for the YIP FITS file.
        grid: Native image pixel grid shared by the PSF products.
        offsets_xy_lod: Off-axis source positions (x, y) in lambda/D, one row
            per PSF in ``psfs``.
        psfs: Off-axis PSF cube, one native image per source offset.
        stellar: Pair of (stellar diameters in lambda/D, stellar intensity maps).
        sky: Sky transmission image on the native grid.
        headers: A common FITS Header or per-product header cards.

    Returns:
        Whatever ``write_yip`` returns for the written file.
    """
    diameters, intensity = stellar
    return write_yip(
        path,
        {
            "grid": grid,
            "sample_kind": "point",
            "headers": headers,
            "offsets_xy_lod": offsets_xy_lod,
            "offaxis_psfs": psfs,
            "stellar_diameters_lod": diameters,
            "stellar_intensity": intensity,
            "sky_transmission": sky,
        },
    )
