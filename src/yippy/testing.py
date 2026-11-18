"""Serialize caller-supplied synthetic coronagraph data for tests."""

from .writer import write_yip


def write_synthetic_yip(path, *, grid, offsets_xy_lod, psfs, stellar, sky, headers):
    """Write fixture arrays; stellar is a pair of (diameters_lod, intensity_maps).

    The grid describes the native image coordinates. Headers can supply a
    common FITS Header or per-product cards, including distinct product grids.
    The caller supplies all radiometry, source coordinates and diameters.
    """
    diameters, intensity = stellar
    return write_yip(
        path,
        {
            "grid": grid,
            "headers": headers,
            "offsets_xy_lod": offsets_xy_lod,
            "offaxis_psfs": psfs,
            "stellar_diameters_lod": diameters,
            "stellar_intensity": intensity,
            "sky_transmission": sky,
        },
    )
