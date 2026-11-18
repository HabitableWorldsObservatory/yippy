"""Write the five native FITS products of a yield input package."""

from collections.abc import Mapping
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
from astropy.io import fits
from optixstuff.grids import Image, Measure

from .header import HeaderData
from .package import YIPPackage
from .schema import PRODUCT_FILES, WRITE_LOCK

_INTENSITY_PRODUCTS = ("offaxis_psfs", "stellar_intensity")
_IMAGE_AXES = {"offaxis_psfs": 3, "stellar_intensity": 3, "sky_transmission": 2}


def yip_header(
    grid,
    *,
    design=None,
    inscribed_diameter_m=None,
    min_wavelength_nm=None,
    max_wavelength_nm=None,
    obscured=None,
    jitter_mas=None,
    n_wavelengths=None,
    n_pointings=None,
    zernike=None,
    wfe_nm=None,
    time_multiplier_detection=None,
    time_multiplier_characterization=None,
):
    """Encode a native focal grid and its optical metadata as FITS cards."""
    if (
        grid.unit != "lod"
        or grid.frame is None
        or grid.frame.diameter_convention != "circumscribed"
    ):
        raise ValueError("YIP image grids require circumscribed lambda/D units")
    header = fits.Header()
    for key, value, comment in (
        ("DESIGN", design, "optical design"),
        ("D", grid.frame.diameter_m, "circumscribed telescope diameter in m"),
        ("D_INSC", inscribed_diameter_m, "inscribed telescope diameter in m"),
        ("PIXSCALE", grid.step, "lambda/D per pixel"),
        ("LAMBDA", grid.frame.reference_wavelength_nm, "reference wavelength in nm"),
        ("MINLAM", min_wavelength_nm, "shortest bandpass wavelength in nm"),
        ("MAXLAM", max_wavelength_nm, "longest bandpass wavelength in nm"),
        ("XCENTER", grid.center_x + 0.5, "optical axis in x corner counts"),
        ("YCENTER", grid.center_y + 0.5, "optical axis in y corner counts"),
        ("OBSCURED", obscured, "obscured aperture-area fraction"),
        ("JITTER", jitter_mas, "RMS jitter per axis in mas"),
        ("N_LAM", n_wavelengths, "number of wavelength samples"),
        ("N_STAR", n_pointings, "number of stellar samples"),
        ("ZERNIKE", zernike, "aberration set"),
        ("WFE", wfe_nm, "wavefront error in nm"),
        ("TMULDET", time_multiplier_detection, "detection time multiplier"),
        (
            "TMULCHAR",
            time_multiplier_characterization,
            "characterization time multiplier",
        ),
    ):
        if value is not None:
            header[key] = (value, comment)
    return header


def _array_hdu(arrays, product):
    data = arrays[product]
    headers = arrays.get("headers") or {}
    if isinstance(headers, fits.Header) or not PRODUCT_FILES.keys() & headers.keys():
        cards = headers
    else:
        cards = headers.get(product)
    grid = None
    if product in _IMAGE_AXES:
        grid = arrays.get("grid")
        if isinstance(grid, Mapping):
            grid = grid[product]
    declared = (cards or {}).get("SAMPKIND")
    sample_kind = arrays.get("sample_kind", declared)
    if (
        product in _INTENSITY_PRODUCTS
        and declared is not None
        and sample_kind != declared
    ):
        raise ValueError("SAMPKIND conflicts with the declared sample_kind")
    if isinstance(data, Image):
        grid = data.grid
        if product in _INTENSITY_PRODUCTS:
            inferred = "integral" if data.measure == Measure.integral() else "point"
            if data.measure == Measure.intensive() or (
                sample_kind is not None and sample_kind != inferred
            ):
                raise ValueError("Image measure conflicts with YIP intensity sampling")
            sample_kind = inferred
            data = data.integrated()
        elif product == "sky_transmission" and data.measure != Measure.intensive():
            raise ValueError("sky transmission must be intensive")
        data = data.data
    data = np.asarray(data)
    if grid is not None and data.shape[-2:] != grid.shape:
        raise ValueError(f"{product} data shape disagrees with its grid")
    header = fits.Header() if grid is None else yip_header(grid)
    if cards is not None:
        header.update(cards)
    if product in _INTENSITY_PRODUCTS:
        header["SAMPKIND"] = (
            "point" if sample_kind is None else sample_kind,
            "detector intensity sampling",
        )
    return fits.PrimaryHDU(data, header or None)


def _validate_products(directory):
    shapes = {}
    for product, filename in PRODUCT_FILES.items():
        header = fits.getheader(directory / filename)
        axes = header["NAXIS"]
        shapes[product] = tuple(header[f"NAXIS{i}"] for i in range(axes, 0, -1))
        if not all(shapes[product]) or not axes:
            raise ValueError(f"Empty YIP product: {product}")
        if product in _IMAGE_AXES:
            expected = _IMAGE_AXES[product]
            if axes != expected:
                raise ValueError(f"{product} requires {expected} axes")
            grid = HeaderData.from_fits_header(header).pixel_grid()
            if not np.isfinite(
                (
                    grid.step,
                    grid.center_x,
                    grid.center_y,
                    grid.frame.reference_wavelength_nm,
                    grid.frame.diameter_m,
                )
            ).all():
                raise ValueError(f"{product} requires finite grid geometry")
    n_psfs = shapes["offaxis_psfs"][0]
    if shapes["offsets_xy_lod"] not in ((n_psfs,), (n_psfs, 2), (2, n_psfs)):
        raise ValueError("Expected one XY offset per off-axis PSF")
    n_diams = shapes["stellar_intensity"][0]
    if shapes["stellar_diameters_lod"] not in ((n_diams,), (1, n_diams), (n_diams, 1)):
        raise ValueError("Expected one diameter per stellar image")


def _publish(staging, path):
    # Concurrent writers stage privately. Only a complete set can publish.
    lock = path / WRITE_LOCK
    try:
        lock.mkdir()
    except FileExistsError as err:
        raise RuntimeError(f"A YIP write is in progress at {path}") from err
    saved, published = [], []
    try:
        if (path / "sinc-delivery.json").exists():
            raise ValueError("Cannot overwrite a sinc delivery with a five-file YIP")
        for name in PRODUCT_FILES.values():
            target = path / name
            if target.exists():
                if not target.is_file():
                    raise ValueError(f"Not a FITS file: {target}")
                target.replace(lock / name)
                saved.append(name)
            (staging / name).replace(target)
            published.append(name)
    except BaseException:
        for name in published:
            (path / name).unlink()
        for name in saved:
            (lock / name).replace(path / name)
        lock.rmdir()
        raise
    for name in saved:
        (lock / name).unlink()
    lock.rmdir()


def _write_delivery(path, package):
    """Publish the selected delivery and its verified inputs in one rename."""
    package._check_source()
    if path.resolve() == package.source.path:
        return path
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("A versioned delivery requires a new or empty destination")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as staging_dir:
        delivery = Path(staging_dir) / "delivery"
        delivery.mkdir()
        for source in package._inputs:
            target = delivery / source.relative_to(package.source.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(source, target)
        package._check_source()
        delivery.replace(path)
    return path


def write_yip(path: str | Path, package_or_arrays: YIPPackage | Mapping) -> Path:
    """Write a native or versioned YIP and return the package directory.

    Pass a YIPPackage or a mapping with offaxis_psfs, offsets_xy_lod,
    stellar_intensity, stellar_diameters_lod and sky_transmission. Image values
    supply their own grids. For bare arrays, supply a common `grid`, a grid
    mapping keyed by image product, or complete `headers`. Headers may be a
    shared FITS header, a dictionary of cards, or a mapping from product
    names to either. Explicit cards override grid defaults. A 1D offset list
    denotes a radial product. Existing unrelated files are retained.

    Native intensity products record their sampling in SAMPKIND. Raw arrays
    default to point samples times cell area. A typed density Image is
    encoded in that convention and a cell integral Image is stored directly. A versioned
    sinc delivery is copied with its manifest and supporting files to a new
    or empty directory. Reopen a multi-plane copy with the desired product ID.
    """
    path = Path(path)
    if not isinstance(package_or_arrays, (YIPPackage, Mapping)):
        raise TypeError("Expected a YIPPackage or a mapping of named arrays")
    if (
        isinstance(package_or_arrays, YIPPackage)
        and package_or_arrays.product_id is not None
    ):
        return _write_delivery(path, package_or_arrays)
    path.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path) as staging_dir:
        staging = Path(staging_dir)
        if isinstance(package_or_arrays, Mapping):
            for product, filename in PRODUCT_FILES.items():
                _array_hdu(package_or_arrays, product).writeto(staging / filename)
        else:
            package = package_or_arrays
            package._check_source()
            for product, filename in PRODUCT_FILES.items():
                target = staging / filename
                copyfile(package._files[product], target)
                if (
                    product in _INTENSITY_PRODUCTS
                    and "SAMPKIND" not in package.fits_header(product)
                ):
                    fits.setval(
                        target,
                        "SAMPKIND",
                        value="point",
                        comment="detector intensity sampling",
                    )
            package._check_source()
        _validate_products(staging)
        _publish(staging, path)
    return path
