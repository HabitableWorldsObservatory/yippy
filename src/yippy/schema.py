"""Native products and filenames of a yield input package."""

PRODUCT_FILES = {
    "offaxis_psfs": "offax_psf.fits",
    "offsets_xy_lod": "offax_psf_offset_list.fits",
    "stellar_intensity": "stellar_intens.fits",
    "stellar_diameters_lod": "stellar_intens_diam_list.fits",
    "sky_transmission": "sky_trans.fits",
}

# Present only while a writer publishes a set of staged products.
WRITE_LOCK = ".yip-writing"
