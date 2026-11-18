"""Private host-side YIP loading and cache preparation."""

import os
import warnings
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.grids import Image
from lod_unit import lod
from optixstuff.apertures import DEFAULT_APERTURE_RULE, Circular, RecordRule
from optixstuff.performance import (
    CORE_OVERSAMPLE,
    SampledProfile,
    performance_curves,
)
from sinchole import Reason
from tqdm import tqdm

from ._precision import dtype_tag, float_dtype
from ._version import __version__
from .interpolation import SincOptions, blend_read
from .logger import logger
from .offjax import OffJAX
from .performance import _iter_xaxis_positions
from .sky_trans import sky_transmission_image
from .stellar_intens import StellarIntens
from .util import (
    PERFORMANCE_VERSION,
    cache_identity_digest,
    load_coro_performance_from_fits,
    read_performance_cache_key,
    save_coro_performance_to_fits,
)

# Invalidate artifacts using corner counts as pixel-center indices, including
# datacube layouts and nominal peak locations in performance tables.
_SAMPLING_CACHE_VERSION = 2

# Namespace of automatically managed cache entries. Schema 2 keys every entry
# by a digest of its complete calculation identity; entries from earlier
# schemas have different names and are never read.
_CACHE_SCHEMA_VERSION = 2

# Oversampling factor for automatically computed performance tables.
_DEFAULT_OVERSAMPLE = CORE_OVERSAMPLE


def _load_cached_datacube(path: Path) -> np.ndarray | None:
    """The cached PSF datacube at ``path``, or None when it is missing or unreadable.

    A run interrupted while writing the cache leaves a truncated file that
    can never load. Such a file is removed, with a warning, so the caller
    rebuilds the cube instead of failing on every later run.
    """
    if not path.exists():
        return None
    try:
        return np.load(path)
    except (ValueError, EOFError, OSError) as err:
        logger.warning(f"Discarding unreadable PSF datacube cache {path}: {err}")
        path.unlink(missing_ok=True)
        return None


def _save_datacube(path: Path, psfs) -> None:
    """Write the PSF datacube to ``path`` through a temporary file.

    The file is renamed into place only once it is complete, so an
    interrupted write never leaves a truncated cache behind.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            np.save(fh, np.asarray(psfs))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class _YIPData:
    """Load native FITS products and prepare immutable query tables."""

    def __init__(
        self,
        package,
        x_symmetric: bool = True,
        y_symmetric: bool = True,
        use_quarter_psf_datacube: bool = True,
        downsample_shape: tuple[int, int] | None = None,
        rule=DEFAULT_APERTURE_RULE,
        leakage_rule: str = "aperture_mean",
        *,
        interpolation: str = "blend",
        sinc_options: SincOptions | None = None,
    ):
        """Prepare interpolation backends and aperture records from native samples."""
        from .sinc_routing import route_interpolation

        self.package = package
        yip_path = package.source.path
        if package.product_id is not None:
            sinc_options = sinc_options or SincOptions()
            if sinc_options.product_id not in (None, package.product_id):
                raise ValueError(
                    "sinc_options.product_id differs from the open package"
                )
            if sinc_options.product_id is None and len(package._manifest["planes"]) > 1:
                sinc_options = replace(sinc_options, product_id=package.product_id)
        delivery, backend, self.interpolation_info = route_interpolation(
            yip_path, interpolation, sinc_options, downsample_shape, package=package
        )
        self._sinc_delivery = delivery
        self.x_symmetric = x_symmetric
        self.y_symmetric = y_symmetric

        logger.info(f"Creating {yip_path.stem} coronagraph")

        self.name = yip_path.stem

        # Store performance curve parameters
        self.rule = rule
        self.leakage_rule = leakage_rule
        self.curves_by_rule = {}

        self.header = package.header.stellar_intensity
        self.pixel_scale_arcsec = self.header.pixscale
        self.frac_obscured = self.header.obscured
        self.stellar_intens = StellarIntens(package)
        self.offaxis_header = (
            delivery.header if delivery else package.header.offaxis_psfs
        )
        if delivery is None and (
            self.offaxis_header.pixel_grid() != self.header.pixel_grid()
        ):
            raise ValueError(
                f"{yip_path.stem}: the off-axis PSF grid differs from the stellar "
                "intensity grid; a YIP stores both on one native grid"
            )

        # Offaxis PSF of the planet as function of separation from the star
        self.offax = (
            backend
            if backend is not None
            else OffJAX(
                package.offaxis_psfs,
                package.offsets_xy_lod,
                self.offaxis_header.pixel_grid(),
                x_symmetric,
                y_symmetric,
                downsample_shape=downsample_shape,
            )
        )

        self.grid = self.offax.grid

        if delivery is not None:
            self.stellar_header = self.header
            self.stellar_pixel_scale_arcsec = self.header.pixscale
            self.offaxis_header = delivery.header
            self.pixel_scale_arcsec = self.offax.pixel_scale_arcsec
            self.sky_header = package.header.sky_transmission

        # Update pixel_scale_arcsec if downsampling was applied
        if downsample_shape is not None:
            self.pixel_scale_arcsec = self.offax.pixel_scale_arcsec

        # Check the realized sampling once, including when curves load from cache.
        pixel_scale_lod = float(self.pixel_scale_arcsec.value)
        if pixel_scale_lod > 0.5:
            warnings.warn(
                f"PSF pixel scale {pixel_scale_lod:g} lambda/D per pixel exceeds "
                "the Nyquist limit of 0.5 lambda/D per pixel; band-limited "
                "aperture measurements may be aliased. Computing anyway.",
                UserWarning,
                stacklevel=2,
            )

        # Get the sky_trans mask
        self.sky_trans = sky_transmission_image(package)

        # PSF datacube axes are (source_y, source_x, y, x).
        # Given the computational cost of generating this
        # datacube, it is only generated when needed.
        self.has_psf_datacube = False
        self.use_quarter_psf_datacube = use_quarter_psf_datacube

        # Shape of the images in the PSFs
        # Use actual PSF shape from offax (accounts for downsampling)
        self.psf_shape = np.array(self.offax.psf_shape)
        assert self.psf_shape[0] == self.psf_shape[1], "PSF must be square"
        self.npixels = self.psf_shape[0]

        self._load_or_compute_performance()
        self._initialize_profiles()

        logger.info(f"Created {yip_path.stem}")

    def create_psf(self, x_lod, y_lod):
        """Return the off-axis photon fraction per pixel at one source position."""
        if self._sinc_delivery is None and self.interpolation_info.requested == "auto":
            return Image(
                self.create_psf_with_info(x_lod, y_lod).intensity,
                self.grid,
                (),
                "per_pixel",
                "stellar_flux_fraction",
                reconstruction="band_limited",
            )
        return self.offax.create_psf(x_lod, y_lod)

    def create_psfs(self, x, y):
        """Return off-axis photon fractions for equal-length coordinate vectors."""
        if self._sinc_delivery is None and self.interpolation_info.requested == "auto":
            return Image(
                self.create_psfs_with_info(x, y).intensity,
                self.grid,
                ("source",),
                "per_pixel",
                "stellar_flux_fraction",
                reconstruction="band_limited",
            )
        return self.offax.create_psfs(x, y)

    def create_psf_with_info(self, x_lod, y_lod):
        """Return an image and backend diagnostics in photon fractions per pixel."""
        if hasattr(self.offax, "create_psf_with_info"):
            return self.offax.create_psf_with_info(x_lod, y_lod)
        finite = jnp.isfinite(x_lod) & jnp.isfinite(y_lod)
        image = self.offax.create_psf(
            jnp.where(finite, x_lod, 0.0), jnp.where(finite, y_lod, 0.0)
        )
        info = self.interpolation_info
        reasons = (
            Reason.LAYOUT_UNSUPPORTED if "LAYOUT_UNSUPPORTED" in info.reasons else 0
        )
        return blend_read(
            image.data, finite, reasons, fallback=info.requested == "auto"
        )

    def create_psfs_with_info(self, x, y):
        """Return batch images with per-query status and backend selections."""
        if hasattr(self.offax, "create_psfs_with_info"):
            return self.offax.create_psfs_with_info(x, y)
        return jax.vmap(self.create_psf_with_info)(x, y)

    def _load_or_compute_performance(self):
        """Select the current rule's complete record from a validated cache."""
        return self._prepare_record()

    def _initialize_profiles(self):
        """Measure radial transmission on the ancillary product's native grid."""
        self.uniform_sky_profile = SampledProfile.from_uniform_sky(self.sky_trans)

    def _datacube_offsets(self):
        """Source (x, y) offsets on the native grid, optionally its positive quarter.

        Even geometric grids have disjoint quadrants; odd grids share axes.
        Reflection storage requires the optical axis at the geometric midpoint.
        """
        grid = self.grid
        columns, rows = np.arange(grid.shape[1]), np.arange(grid.shape[0])
        if self.use_quarter_psf_datacube:
            if (grid.center_x, grid.center_y) != (
                (grid.shape[1] - 1) / 2,
                (grid.shape[0] - 1) / 2,
            ):
                raise ValueError("quarter datacubes require a geometric grid center")
            columns = columns[columns >= grid.center_x]
            rows = rows[rows >= grid.center_y]
        return np.meshgrid(
            (columns - grid.center_x) * grid.step,
            (rows - grid.center_y) * grid.step,
            indexing="xy",
        )

    def _prepare_dense_psfs(self, batch_size=128):
        """Load the PSF datacube from a file or generate it if it doesn't exist.

        The PSF datacube has axes (source_y, source_x, y, x).
        Given the computational cost of generating this
        datacube, it is only generated when needed and saved to a numpy binary
        file in the yip_path directory.

        Args:
            batch_size (int):
                Number of PSFs to generate in each batch. Default is 128.
        """
        if self._sinc_delivery is not None:
            return self._create_delivery_datacube(batch_size)
        if self.offax.type == "1d":
            logger.warning(
                "Building a PSF datacube from a 1D YIP: off-axis PSFs are "
                "translated to each source position, not rotated to its "
                "position angle, so azimuthal PSF structure keeps its +x "
                "orientation. Use a 2D YIP for spatially resolved scenes."
            )
        backend = jax.default_backend().lower()
        if backend in ("gpu", "tpu"):
            return self._prepare_dense_psfs_gpu(batch_size=batch_size)
        ext = "_quarter" if self.use_quarter_psf_datacube else ""
        datacube_path = self._datacube_cache_path
        cached = _load_cached_datacube(datacube_path)
        if cached is not None:
            logger.info(f"Loading PSF datacube from {datacube_path}.")
            psfs = jnp.asarray(cached)
        else:
            x_lod, y_lod = self._datacube_offsets()
            psfs_shape = (*x_lod.shape, *self.psf_shape)
            psfs = np.zeros(psfs_shape, dtype=float_dtype())
            points = np.column_stack((x_lod.flatten(), y_lod.flatten()))
            n_points = points.shape[0]

            logger.info(
                f"Calculating {'quarter' if ext else 'full'} data cube of "
                f"spatially dependent PSFs ({n_points} points), please hold..."
            )
            with tqdm(total=n_points, desc="Computing PSFs") as pb:
                for i in range(0, n_points, batch_size):
                    # Select the current batch
                    batch_points = points[i : i + batch_size]
                    batch_psfs = self.offax.create_psfs_parallel(
                        batch_points[:, 0], batch_points[:, 1]
                    )

                    # Store the batch in the data cube
                    psfs.reshape((-1, self.npixels, self.npixels))[
                        i : i + batch_size
                    ] = batch_psfs.data
                    pb.update(batch_points.shape[0])
            _save_datacube(datacube_path, psfs)
            logger.info(f"PSF datacube saved to {datacube_path}.")

        # Move datacube to GPU/TPU device if conditions are met
        backend = jax.default_backend().lower()
        if self.use_quarter_psf_datacube and backend in ("gpu", "tpu"):
            # Check if already a JAX array on the target device
            target_device = jax.devices(backend)[0]
            already_on_device = (
                hasattr(psfs, "devices") and target_device in psfs.devices()
            )

            if already_on_device:
                logger.info(f"PSF datacube already on {backend.upper()} device")
            else:
                logger.info(
                    f"Moving PSF datacube to {backend.upper()} device "
                    "(quarter symmetric datacube)"
                )
                # Convert to JAX array and place on device
                # Note: avoid jnp.array() on existing JAX array to prevent copy
                try:
                    if not isinstance(psfs, jax.Array):
                        psfs = jnp.asarray(psfs)
                    psfs = jax.device_put(psfs, target_device)
                    logger.info(
                        f"Successfully moved PSF datacube to {backend.upper()} device"
                    )
                except (MemoryError, RuntimeError) as e:
                    logger.warning(
                        f"Failed to move PSF datacube to {backend.upper()} "
                        f"device (insufficient memory): {e}. Keeping on CPU."
                    )
                    # psfs remains as numpy array on CPU

        self.has_psf_datacube = True
        self.psf_datacube = psfs

    def _create_delivery_datacube(self, batch_size):
        """Cache midpoint-grid reads with integrity-bound status selections."""
        import json

        from .sinc_delivery import file_sha256

        path = self._datacube_cache_path
        status_path = path.with_suffix(".status.npz")
        integrity_path = path.with_suffix(".status.sha256")
        expected = cache_identity_digest(self._datacube_identity())
        cached = _load_cached_datacube(path)
        if (
            cached is not None
            and status_path.exists()
            and integrity_path.exists()
            and integrity_path.read_text() == file_sha256(status_path)
        ):
            try:
                with np.load(status_path) as data:
                    valid = (
                        int(data["schema"]) == 1
                        and str(data["identity"]) == expected
                        and str(data["cube_sha256"]) == file_sha256(path)
                    )
                    if valid:
                        self.psf_datacube = jnp.asarray(cached)
                        self.psf_datacube_info = {
                            key: data[key]
                            for key in (
                                "selected_backend",
                                "status",
                                "reason_bits",
                                "accepted",
                            )
                        }
                        self.has_psf_datacube = True
                        return
            except (ValueError, EOFError, OSError, KeyError):
                pass
        xx, yy = self._datacube_offsets()
        x, y = xx.ravel(), yy.ravel()
        images = []
        diagnostics = {
            key: [] for key in ("selected_backend", "status", "reason_bits", "accepted")
        }
        for start in range(0, len(x), batch_size):
            result = self.create_psfs_with_info(
                x[start : start + batch_size], y[start : start + batch_size]
            )
            images.append(np.asarray(result.intensity))
            for key in diagnostics:
                diagnostics[key].append(np.asarray(getattr(result, key)))
        cube = np.concatenate(images).reshape((*xx.shape, *self.psf_shape))
        diagnostics = {
            key: np.concatenate(value).reshape(xx.shape)
            for key, value in diagnostics.items()
        }
        _save_datacube(path, cube)
        tmp = status_path.with_name(f"{status_path.name}.{os.getpid()}.tmp")
        try:
            with tmp.open("wb") as stream:
                np.savez(
                    stream,
                    schema=1,
                    identity=expected,
                    cube_sha256=file_sha256(path),
                    query_grid=json.dumps({"x": x.tolist(), "y": y.tolist()}),
                    **diagnostics,
                )
            os.replace(tmp, status_path)
            integrity_path.write_text(file_sha256(status_path))
        finally:
            tmp.unlink(missing_ok=True)
        self.psf_datacube, self.psf_datacube_info = jnp.asarray(cube), diagnostics
        self.has_psf_datacube = True

    def _prepare_dense_psfs_gpu(self, batch_size=128):
        """Generate PSF datacube on GPU/TPU, transferring each batch to host.

        Uses vmap+jit (``offax.create_psfs``) instead of shard_map to avoid
        the CPU multi-device overhead that ``create_psfs_parallel`` carries.
        Each batch is copied to a pre-allocated host buffer immediately after
        synthesis, then ``np.save`` writes the final cube. The on-device
        accumulation + ``jnp.concatenate`` pattern is avoided because it
        silently corrupts output on some GPUs (Blackwell observed; A100 with
        broken CUDA12 plugin observed).

        Args:
            batch_size (int):
                Number of PSFs to generate per batch. Default is 128.
        """
        ext = "_quarter" if self.use_quarter_psf_datacube else ""
        if self._sinc_delivery is not None:
            return self._create_delivery_datacube(batch_size)
        datacube_path = self._datacube_cache_path
        cached = _load_cached_datacube(datacube_path)
        if cached is not None:
            logger.info(f"Loading PSF datacube from {datacube_path}.")
            psfs = jnp.asarray(cached)
        else:
            x_lod, y_lod = self._datacube_offsets()
            psfs_shape = (*x_lod.shape, *self.psf_shape)
            points = np.column_stack((x_lod.flatten(), y_lod.flatten()))
            n_points = points.shape[0]

            backend = jax.default_backend().upper()
            logger.info(
                f"Generating {'quarter' if ext else 'full'} PSF datacube "
                f"on {backend} ({n_points} points)..."
            )

            # Pre-allocate host buffer and transfer each batch immediately.
            # Avoids a silent jnp.concatenate corruption observed on some GPUs
            # (e.g. Blackwell / RTX PRO 6000) when aggregating many GPU buffers
            # before save.
            psfs = np.empty(psfs_shape, dtype=float_dtype())
            flat_view = psfs.reshape(-1, self.npixels, self.npixels)
            with tqdm(total=n_points, desc="Computing PSFs") as pb:
                for i in range(0, n_points, batch_size):
                    batch_x = jnp.array(points[i : i + batch_size, 0])
                    batch_y = jnp.array(points[i : i + batch_size, 1])
                    batch_psfs = self.offax.create_psfs(batch_x, batch_y)
                    flat_view[i : i + batch_size] = np.asarray(batch_psfs.data)
                    pb.update(batch_x.shape[0])

            _save_datacube(datacube_path, psfs)
            logger.info(f"PSF datacube saved to {datacube_path}.")

        target_device = jax.devices()[0]
        if not (hasattr(psfs, "devices") and target_device in psfs.devices()):
            psfs = jax.device_put(jnp.asarray(psfs), target_device)
            logger.info(f"PSF datacube on {jax.default_backend().upper()} device")

        self.has_psf_datacube = True
        self.psf_datacube = psfs

    def __repr__(self):
        """String representation of the Coronagraph object."""
        base_str = f"Coronagraph {self.name} ({self.package.source.path})\n"
        base_str += f"{self.offax.type} off-axis PSFs, {self.offax.n_psfs} provided"

        if hasattr(self, "performance"):
            base_str += f"\nAperture: {self.rule}; leakage: {self.leakage_rule}"
            base_str += f"\nInner working angle: {self.performance.iwa_lod:.2f} lod"
            base_str += f"\nData half-width: {self.performance.data_extent_lod:.2f} lod"

        if self.has_psf_datacube:
            base_str += "\nPSF datacube loaded"

        return base_str

    @property
    def _cache_dir(self) -> Path:
        """Directory for all yippy-computed artifacts."""
        d = self.package.source.path / "yippy_cache"
        d.mkdir(exist_ok=True)
        return d

    def _cache_identity_base(self, artifact: str) -> dict:
        """Identity shared by artifacts computed from this source generation."""
        identity = {
            "artifact": artifact,
            "schema": _CACHE_SCHEMA_VERSION,
            "sampling": _SAMPLING_CACHE_VERSION,
            "source_signature": self.package.source.digest,
            "npixels": int(self.npixels),
            "dtype": dtype_tag(),
        }
        info = getattr(self, "interpolation_info", None)
        if info is not None and (info.manifest_sha256 or info.requested != "blend"):
            from .sinc_routing import cache_interpolation_identity

            identity["interpolation"] = cache_interpolation_identity(info, self.offax)
        return identity

    def _datacube_identity(self) -> dict:
        """Complete identity of the PSF datacube this object would store.

        The physical symmetry flags only change PSFs synthesized at negative
        offsets, which a quarter cube never evaluates, so they are part of the
        identity of full cubes only.
        """
        identity = self._cache_identity_base("psf_datacube")
        identity["layout"] = 2
        identity["grid"] = {
            "center_xy": (self.grid.center_x, self.grid.center_y),
            "basis": self.grid.basis,
        }
        identity["storage"] = "quarter" if self.use_quarter_psf_datacube else "full"
        if not self.use_quarter_psf_datacube:
            identity["x_symmetric"] = bool(self.x_symmetric)
            identity["y_symmetric"] = bool(self.y_symmetric)
        return identity

    @property
    def _datacube_cache_path(self) -> Path:
        """PSF datacube cache file.

        The name carries quarter/full, active float dtype, the realized PSF
        pixel count, the source signature, and a digest of the complete
        identity (see ``_datacube_identity``), which also covers the selected
        source files, symmetry flags of full cubes, the sampling implementation
        version, and the cache schema.
        """
        ext = "_quarter" if self.use_quarter_psf_datacube else ""
        digest = cache_identity_digest(self._datacube_identity())[:16]
        return (
            self._cache_dir / f"psf_datacube{ext}_{dtype_tag()}_{self.npixels}px_"
            f"{self.package.source.digest}_{digest}.npy"
        )

    @property
    def _perf_dir(self) -> Path:
        """Directory for cached performance curve FITS files."""
        d = self._cache_dir / "performance"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _perf_identity(
        self, rule, leakage_rule, stellar_diameter, separations_lod=None
    ):
        """Complete identity of a weighted photometry record."""
        from dataclasses import asdict

        identity = self._cache_identity_base("performance")
        identity.update(
            {
                "performance_record": PERFORMANCE_VERSION,
                "aperture_algorithm": "continuum_region_dft_peak_v1",
                "yippy_version": __version__,
                "table_storage": "float64",
                "rule": {"name": type(rule).__name__, **asdict(rule)},
                "leakage_rule": leakage_rule,
                "stellar_diameter": stellar_diameter,
                "oversample": CORE_OVERSAMPLE,
                "separations_lod": None
                if separations_lod is None
                else np.asarray(separations_lod).tolist(),
                "offaxis_grid": asdict(self.grid),
                "stellar_grid": asdict(self.stellar_intens.grid),
            }
        )
        return identity

    def _auto_perf_identity(self):
        return self._perf_identity(
            self.rule, self.leakage_rule, float(self.stellar_intens.diams[0].value)
        )

    def _perf_filename(self):
        digest = cache_identity_digest(self._auto_perf_identity())[:16]
        return (
            f"perf_{type(self.rule).__name__}_{self.npixels}px_{dtype_tag()}_"
            f"{self.package.source.digest}_{digest}.fits"
        )

    def _prepare_record(
        self,
        rule=None,
        stellar_diameter=None,
        *,
        separations_lod=None,
        leakage_rule=None,
        save_to_fits=True,
        performance_file=None,
        load_from_file=None,
        cache_dir=None,
    ):
        """Build/select one complete record; explicitly imported metadata must match.

        Persistent records store all photometry fields on one axis, plus aperture,
        averaging rule, stellar diameter and frame. Corrupt automatic entries
        are cache misses. Explicit imports fail clearly instead of silently
        overriding their physical definitions.
        """
        rule = self.rule if rule is None else rule
        if not isinstance(rule, RecordRule):
            raise TypeError("Performance records require Circular or HalfMaximum")
        leakage_rule = self.leakage_rule if leakage_rule is None else leakage_rule
        diameter = (
            float(self.stellar_intens.diams[0].value)
            if stellar_diameter is None
            else float(getattr(stellar_diameter, "value", stellar_diameter))
        )
        identity = self._perf_identity(rule, leakage_rule, diameter, separations_lod)
        expected = cache_identity_digest(identity)
        directory = self._perf_dir if cache_dir is None else Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        filename = performance_file or (
            f"perf_{type(rule).__name__}_{self.npixels}px_{dtype_tag()}_"
            f"{self.package.source.digest}_{expected[:16]}.fits"
        )
        path = directory / (load_from_file or filename)
        record = None
        automatic = load_from_file is None
        if not automatic or (
            path.exists() and read_performance_cache_key(path) == expected
        ):
            try:
                record = load_coro_performance_from_fits(path.name, path.parent)
                if (
                    record.rule != rule
                    or record.leakage_rule != leakage_rule
                    or record.frame != self.grid.frame
                    or record.stellar_diameter != diameter
                ):
                    raise ValueError(
                        "performance record rule, leakage_rule, frame "
                        "or stellar diameter mismatch"
                    )
                if separations_lod is not None and not np.array_equal(
                    record.separation_lod, separations_lod
                ):
                    raise ValueError("performance record separation axis mismatch")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                if not automatic:
                    raise ValueError(
                        f"Cannot import performance record: {exc}"
                    ) from exc
                logger.warning(f"Ignoring incompatible performance cache {path}: {exc}")
                record = None
        if record is None:
            positions = list(_iter_xaxis_positions(self))
            if isinstance(rule, Circular):
                positions = [
                    p
                    for p in positions
                    if 0 <= p.px < p.psf.shape[1] and 0 <= p.py < p.psf.shape[0]
                ]
            samples = {
                p.separation: Image(
                    p.psf,
                    self.grid,
                    (),
                    "per_pixel",
                    "stellar_flux_fraction",
                    reconstruction="band_limited",
                )
                for p in positions
            }
            native_axis = np.array([p.separation for p in positions])
            locations = [
                (
                    (p.px - self.grid.center_x) * self.grid.step,
                    (p.py - self.grid.center_y) * self.grid.step,
                )
                for p in positions
            ]
            record = performance_curves(
                (
                    samples.__getitem__
                    if separations_lod is None
                    else lambda s: self.create_psf(s, 0.0)
                ),
                self.stellar_intens(diameter * lod),
                rule,
                separations_lod=native_axis
                if separations_lod is None
                else separations_lod,
                source_positions_lod=locations if separations_lod is None else None,
                leakage_rule=leakage_rule,
                stellar_diameter=diameter,
            )
            if save_to_fits:
                save_coro_performance_to_fits(
                    record, filename, directory, identity=identity
                )
        self.rule, self.leakage_rule = rule, leakage_rule
        self.performance = record
        self.curves_by_rule[rule] = record
        return record
