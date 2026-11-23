"""Native products and source identity of a Yield Input Package."""

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from weakref import finalize

import numpy as np
from astropy.io import fits
from optixstuff.apertures import DEFAULT_APERTURE_RULE

from .datasets import fetch_yip, yip_exists
from .delivery.format import delivery_header, detect_delivery, select_plane
from .header import HeaderData
from .schema import PRODUCT_FILES, WRITE_LOCK


@dataclass(frozen=True)
class Source:
    """Directory and SHA-256 identity of the selected package inputs.

    Attributes:
        path: Directory holding the package inputs.
        digest: SHA-256 hex digest of the product names, relative file names and file
            contents.
    """

    path: Path
    digest: str


def _generation(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class _SourceFile:
    """An open source generation shared by hashing, header intake and lazy reads.

    Args:
        path: Path of the file to open for binary reading.
    """

    def __init__(self, path):
        self.stream = path.open("rb")
        finalize(self, self.stream.close)
        self.lock = Lock()
        self.generation = _generation(os.fstat(self.stream.fileno()))
        self.digest = self._content_digest()

    def _content_digest(self):
        self.stream.seek(0)
        return hashlib.file_digest(self.stream, "sha256").digest()

    def matches(self, path):
        current = _generation(path.stat())
        if current == self.generation:
            return True
        # A replacement inode belongs to a new open, even at the same pathname.
        if current[:3] != self.generation[:3]:
            return False
        with self.lock:
            if self._content_digest() != self.digest or current != _generation(
                path.stat()
            ):
                return False
            self.generation = current
            return True

    def fits(self, reader):
        with self.lock:
            self.stream.seek(0)
            return reader(self.stream, 0, memmap=False)


def _source(path, products, files, product_id):
    digest = hashlib.sha256()
    if product_id is not None:
        digest.update(product_id.encode() + b"\0")
    for product, file in sorted(products.items()):
        digest.update(
            product.encode()
            + b"\0"
            + file.relative_to(path).as_posix().encode()
            + b"\0"
        )
    for file, snapshot in files.items():
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(snapshot.digest)
    return Source(path, digest.hexdigest())


@dataclass(frozen=True)
class ProductHeaders:
    """Metadata and native pixel geometry for each sampled image product.

    Attributes:
        offaxis_psfs: Header of the off-axis PSF cube.
        stellar_intensity: Header of the stellar intensity cube.
        sky_transmission: Header of the sky transmission image.
    """

    offaxis_psfs: HeaderData
    stellar_intensity: HeaderData
    sky_transmission: HeaderData


@dataclass(frozen=True, eq=False)
class YIPPackage:
    """The measured samples in shipped order, before interpolation or symmetry.

    Attributes:
        source: Directory and content digest of the selected inputs.
        product_id: Selected plane ID of a versioned delivery, or None for a native
            package.
    """

    source: Source
    product_id: str | None
    _files: Mapping[str, Path]
    _inputs: Mapping[Path, _SourceFile]
    _header_cards: Mapping[str, str]
    _manifest_json: str | None

    @property
    def _manifest(self):
        return (
            json.loads(self._manifest_json) if self._manifest_json is not None else None
        )

    def _check_source(self):
        """Validate the original generation without reopening any source files."""
        try:
            unchanged = all(
                snapshot.matches(path) for path, snapshot in self._inputs.items()
            )
        except FileNotFoundError:
            unchanged = False
        if not unchanged or (self.source.path / WRITE_LOCK).exists():
            raise ValueError("YIP source files changed; open the package again")

    def fits_header(self, product):
        """Return an independent FITS header, including comments and extra cards.

        Args:
            product: Native product name, such as ``"offaxis_psfs"``.

        Returns:
            A new FITS header with the cards of the product file.
        """
        return fits.Header.fromstring(self._header_cards[product], sep="\n")

    @cached_property
    def header(self):
        """Unit-aware image headers, each with its own pixel grid.

        Returns:
            Headers of the three image products, each with units attached.
        """
        manifest = self._manifest
        offaxis = self.fits_header("offaxis_psfs")
        return ProductHeaders(
            HeaderData.from_fits_header(offaxis)
            if manifest is None
            else delivery_header(
                offaxis, manifest, select_plane(manifest, self.product_id)
            ),
            HeaderData.from_fits_header(self.fits_header("stellar_intensity")),
            HeaderData.from_fits_header(self.fits_header("sky_transmission")),
        )

    @cached_property
    def _arrays(self):
        return {}

    def _native_array(self, product):
        """Decode a FITS array once, retaining its original shape and precision."""
        if product not in self._arrays:
            self._check_source()
            data = self._inputs[self._files[product]].fits(fits.getdata)
            # Immutable bytes also prevent writes through a view's base array.
            data = np.frombuffer(data.tobytes(), dtype=data.dtype).reshape(data.shape)
            self._check_source()
            self._arrays[product] = data
        return self._arrays[product]

    @cached_property
    def offsets_xy_lod(self):
        """Native (x, y) offsets in lambda/D, one row per shipped PSF.

        Returns:
            Native (x, y) offsets in lambda/D, shape [n_psf, 2]; a 1D offset list is
            returned with y equal to zero.
        """
        raw = self._native_array("offsets_xy_lod")
        count = self.header.offaxis_psfs.naxis3
        if raw.ndim == 1:
            offsets = np.column_stack((raw, np.zeros_like(raw)))
            offsets = np.frombuffer(offsets.tobytes(), dtype=offsets.dtype).reshape(
                -1, 2
            )
        elif raw.shape == (2, count) and count != 2:
            offsets = raw.T
        else:
            offsets = raw
        if offsets.shape != (count, 2):
            raise ValueError("Expected one XY offset per off-axis PSF")
        return offsets

    @property
    def stellar_diameters_lod(self):
        """Delivered angular diameters in lambda/D."""
        return self._native_array("stellar_diameters_lod").reshape(-1)

    @property
    def offaxis_psfs(self):
        """Native off-axis PSFs with axes (source, y, x)."""
        return self._native_array("offaxis_psfs")

    @property
    def stellar_intensity(self):
        """Native stellar intensity with axes (diameter, y, x)."""
        return self._native_array("stellar_intensity")

    @property
    def sky_transmission(self):
        """Native dimensionless transmission with axes (y, x)."""
        return self._native_array("sky_transmission")

    def prepare(self, rules=(DEFAULT_APERTURE_RULE,), **options):
        """Prepare the numerical coronagraph from this package.

        Args:
            rules: Aperture rules to prepare records for.
            **options: Further keyword arguments of ``Coronagraph``.

        Returns:
            The numerical ``Coronagraph`` prepared from this package.
        """
        from .coronagraph import Coronagraph

        return Coronagraph(self, rules=rules, **options)


def open_yip(path_or_name, *, product_id=None, filenames=None) -> YIPPackage:
    """Open a directory or catalog name, reading image samples on first access.

    For a multi-plane sinc delivery, select the monochromatic product by ID.
    A filename override maps native product names to local FITS filenames.

    Args:
        path_or_name: Package directory, or the catalog name of a YIP to fetch when no
            such directory exists.
        product_id: Plane ID to select in a multi-plane versioned delivery; it must be
            None for a native package.
        filenames: Optional mapping from native product names to local FITS filenames
            that override the defaults.

    Returns:
        The open package; image samples are read on first access.
    """
    path = Path(path_or_name).expanduser()
    if isinstance(path_or_name, str) and not path.exists() and yip_exists(path_or_name):
        path = Path(fetch_yip(path_or_name))
    root = path.resolve(strict=True)
    if (root / WRITE_LOCK).exists():
        raise ValueError("YIP source files changed during publication")
    manifest_file = root / "sinc-delivery.json"
    manifest_generation = (
        _generation(manifest_file.stat()) if manifest_file.exists() else None
    )
    manifest = detect_delivery(root)
    if product_id is not None and manifest is None:
        raise ValueError("product_id selects a plane in a versioned delivery")
    directory = root if manifest is None else root / manifest["legacy_package_path"]
    names = dict(PRODUCT_FILES)
    if filenames:
        if filenames.keys() - names.keys():
            raise ValueError("Unknown YIP product in filename overrides")
        names.update(filenames)
    files = {name: directory / filename for name, filename in names.items()}
    inputs = set(files.values())
    if manifest is not None:
        # A versioned delivery may also ship a complete legacy comparison view.
        legacy_inputs = {file for file in files.values() if file.is_file()}
        planes = manifest["planes"]
        plane = select_plane(manifest, product_id)
        product_id = plane["id"]
        files["offaxis_psfs"] = root / plane["psf_path"]
        files["offsets_xy_lod"] = root / plane["offsets_path"]
        inputs = set(files.values()) | legacy_inputs
        inputs.add(manifest_file)
        inputs.add(root / manifest["training_coordinates_path"])
        inputs.update(
            root / name
            for name in manifest.get("extensions", {}).get("file_sha256", {})
        )
        for entry in [
            *planes,
            *manifest.get("checks", ()),
            *manifest.get("mirror_checks", ()),
            manifest.get("reflection") or {},
            manifest.get("validation") or {},
            manifest.get("stamp_validation") or {},
        ]:
            inputs.update(
                root / value
                for key, value in entry.items()
                if key.endswith("_path") and value
            )
    for file in inputs:
        if not file.resolve().is_relative_to(root):
            raise ValueError(f"YIP product outside its package directory: {file.name}")
    snapshots = {file: _SourceFile(file) for file in sorted(inputs)}
    manifest_snapshot = snapshots.get(manifest_file)
    if (
        manifest_snapshot.generation if manifest_snapshot else None
    ) != manifest_generation:
        raise ValueError("YIP manifest changed during package open")
    source = _source(root, files, snapshots, product_id)
    cards = {
        name: snapshots[file].fits(fits.getheader).tostring(sep="\n", padding=False)
        for name, file in files.items()
    }
    package = YIPPackage(
        source,
        product_id,
        MappingProxyType(files),
        MappingProxyType(snapshots),
        MappingProxyType(cards),
        json.dumps(manifest) if manifest is not None else None,
    )
    package._check_source()
    return package
