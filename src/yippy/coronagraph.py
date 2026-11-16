"""Base coronagraph class."""

import hashlib
import os
import warnings
from pathlib import Path

import astropy.io.fits as pyfits
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.grids import Image
from lod_unit import lod
from tqdm import tqdm

from ._precision import dtype_tag, float_dtype
from ._version import __version__
from .apertures import Circular
from .export import export_ayo_csv
from .header import HeaderData
from .interpolation import SincOptions, blend_read
from .logger import logger
from .offjax import OffJAX
from .performance import (
    CORE_OVERSAMPLE,
    PerformanceCurves,
    compute_core_area_curve,
    compute_core_mean_intensity_curve,
    compute_occ_trans_curve,
    compute_radial_average,
    compute_raw_contrast_curve,
    compute_throughput_curve,
    core_mean_intensity_profile,
    performance_curves,
    plot_performance_curve,
)
from .sky_trans import SkyTrans
from .stellar_intens import StellarIntens
from .util import (
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


class Coronagraph:
    """Primary object for simulating a coronagraph.

    The Coronagraph object manages the coronagraph response for both on-axis
    and off-axis sources. It is primarily called with either
    `:ref:pydata:Coronagraph.offax(x,y)`, to get the off-axis response at a
    given (x,y) offset from the star, or `Coronagraph.stellar(r)` to get the
    coronagraph response at a given stellar angular diameter r.
    """

    def __init__(
        self,
        yip_path: Path,
        stellar_intens_file: str = "stellar_intens.fits",
        stellar_diam_file: str = "stellar_intens_diam_list.fits",
        offax_data_file: str = "offax_psf.fits",
        offax_offsets_file: str = "offax_psf_offset_list.fits",
        sky_trans_file: str = "sky_trans.fits",
        x_symmetric: bool = True,
        y_symmetric: bool = True,
        use_quarter_psf_datacube: bool = True,
        downsample_shape: tuple[int, int] | None = None,
        rule=Circular(0.7),  # noqa: B008 - frozen rule
        contrast_floor: float | None = None,
        leakage_rule: str = "aperture_mean",
        *,
        interpolation: str = "blend",
        sinc_options: SincOptions | None = None,
    ):
        """Initialize the Coronagraph object.

        Loads the coronagraph data from the given yield input package and creates
        the interpolation functions for the stellar intensity and off-axis PSF.

        Args:
            yip_path (Path):
                Yield input package directory. Must have fits files
                    offax_psf_offset_list - The off-axis PSF list
                    offax_psf - PSF of off-axis sources
                    sky_trans - Sky transmission data
            stellar_intens_file (str):
                Name of the stellar intensity file. Default is stellar_intens.fits
            stellar_diam_file (str):
                Name of the stellar intensity diameter list file. Default is
                stellar_intens_diam_list.fits
            offax_data_file (str):
                Name of the off-axis PSF file. Default is offax_psf.fits
            offax_offsets_file (str):
                Name of the off-axis PSF offset list file. Default is
                offax_psf_offset_list.fits
            sky_trans_file (str):
                Name of the sky transmission data file. Default is sky_trans.fits.
            performance_file (str):
                Name of the coronagraph performance (contrast, throughput) fits file.
            x_symmetric (bool):
                Whether off-axis PSFs are symmetric about the x-axis. Default is True.
            y_symmetric (bool):
                Whether off-axis PSFs are symmetric about the y-axis. Default is True.
            use_quarter_psf_datacube (bool):
                Whether to compute the PSF datacube in only the first quadrant.
                This is faster and uses less memory, but may not be accurate for
                all coronagraphs. Default is False.
            downsample_shape (tuple[int, int] | None):
                Optional target shape (ny, nx) to downsample PSFs to. If provided,
                all PSFs will be resampled to this shape immediately after loading,
                conserving total flux. The pixel_scale_arcsec will be updated
                accordingly.
                Default is None (no downsampling).
            rule: Static Circular, HalfMaximum or Matched photometric rule.
            leakage_rule: aperture_mean or azimuthal_ring_mean for the density.
            contrast_floor: Optional floor applied by the contrast accessor;
                the physical record and cache always retain unfloored values.
            interpolation (str):
                Off-axis backend: blend (default), auto, or strict sinc. Legacy
                blend loading and caches are unchanged. Versioned quadrant
                deliveries require sinc; they have no blend representation.
            sinc_options (SincOptions | None):
                Selected fixed product, permission restriction, allocation
                ceilings and read policy. Error budgets use photons per pixel.
        """
        ###################
        # Read input data #
        ###################
        yip_path = Path(yip_path)
        self.yip_path = yip_path
        from .sinc_routing import route_interpolation

        delivery, backend, self.interpolation_info = route_interpolation(
            yip_path, interpolation, sinc_options, downsample_shape
        )
        self._sinc_options = sinc_options or SincOptions()
        self._sinc_delivery = delivery
        package_path = (
            yip_path
            if delivery is None
            else yip_path / delivery.manifest["legacy_package_path"]
        )

        # Capture the source generation before reading any data, so cache
        # entries written later by this object are labeled with the files it
        # actually loaded even if the package is edited in the meantime.
        self._source_sig = self._source_signature()
        self._source_files = {
            "stellar_intens": stellar_intens_file,
            "stellar_diams": stellar_diam_file,
            "offax_psf": offax_data_file,
            "offax_offsets": offax_offsets_file,
        }
        self.x_symmetric = x_symmetric
        self.y_symmetric = y_symmetric

        logger.info(f"Creating {yip_path.stem} coronagraph")

        self.name = yip_path.stem

        # Store performance curve parameters
        self.rule = rule
        self.leakage_rule = leakage_rule
        self.contrast_floor = contrast_floor
        self.curves_by_rule = {}

        # Get header and calculate the lambda/D value
        stellar_intens_header = pyfits.getheader(
            Path(package_path, stellar_intens_file), 0
        )

        # Get pixel scale with units
        self.header = HeaderData.from_fits_header(stellar_intens_header)
        self.pixel_scale_arcsec = self.header.pixscale
        self.frac_obscured = self.header.obscured

        # Stellar intensity of the star being observed as function of stellar
        # angular diameter (unitless)
        self.stellar_intens = StellarIntens(
            package_path, stellar_intens_file, stellar_diam_file
        )

        # Offaxis PSF of the planet as function of separation from the star
        self.offax = (
            backend
            if backend is not None
            else OffJAX(
                yip_path,
                offax_data_file,
                offax_offsets_file,
                self.pixel_scale_arcsec,
                x_symmetric,
                y_symmetric,
                downsample_shape=downsample_shape,
                grid=self.header.pixel_grid(),
            )
        )

        self.grid = self.offax.grid

        if delivery is not None:
            self.stellar_header = self.header
            self.stellar_pixel_scale_arcsec = self.header.pixscale
            self.offaxis_header = delivery.header
            self.pixel_scale_arcsec = self.offax.pixel_scale_arcsec
            self.sky_header = HeaderData.from_fits_header(
                pyfits.getheader(package_path / sky_trans_file)
            )

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
        self.sky_trans = SkyTrans(package_path, sky_trans_file)

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
            (1 << 21)
            if "EXTRA_MISSING" in info.reasons
            else ((1 << 0) if "LAYOUT_UNSUPPORTED" in info.reasons else 0)
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
        return self.compute_all_performance_curves()

    @property
    def iwa_lod(self):
        """Half-maximum-throughput inner working angle of the selected rule."""
        return self.performance.iwa_lod

    @property
    def data_extent_lod(self):
        """Array half-width; no outer working angle is inferred from data support."""
        return self.performance.data_extent_lod

    def _profile_record(self, separation, name, values):
        """Put ancillary sampled quantities through the common curve evaluator."""
        base = self.performance
        fields = {
            key: base.evaluate(key, separation)
            for key in (
                "throughput",
                "core_area_lod2",
                "leakage_fraction",
                "leakage_density_lod2",
                "raw_contrast",
            )
        }
        return PerformanceCurves(
            separation,
            base.frame,
            base.rule,
            **fields,
            iwa_lod=base.iwa_lod,
            data_extent_lod=base.data_extent_lod,
            leakage_rule=base.leakage_rule,
            extra={name: values},
        )

    def _initialize_profiles(self):
        sep, values = compute_occ_trans_curve(self)
        self._occultation_record = self._profile_record(
            sep, "occulter_transmission", values
        )
        sep, profiles = compute_core_mean_intensity_curve(self)
        self.core_intensity_dict = profiles
        self.core_intensity_separations = sep
        self._core_intensity_cache = {}
        self._core_intensity_interpolation = "keys"
        self._core_intensity_diameter = float(self.stellar_intens.diams[0].value)
        self._intensity_record = self._profile_record(
            sep, "core_mean_intensity", profiles[self.stellar_intens.diams[0]]
        )

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

    def create_psf_datacube(self, batch_size=128):
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
            return self._create_psf_datacube_gpu(batch_size=batch_size)
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

    def _create_psf_datacube_gpu(self, batch_size=128):
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
        base_str = f"Coronagraph {self.name} ({self.yip_path})\n"
        base_str += f"{self.offax.type} off-axis PSFs, {self.offax.n_psfs} provided"

        if hasattr(self, "performance"):
            base_str += f"\nAperture: {self.rule}; leakage: {self.leakage_rule}"
            base_str += f"\nInner working angle: {self.iwa_lod:.2f} lod"
            base_str += f"\nData half-width: {self.data_extent_lod:.2f} lod"

        if self.has_psf_datacube:
            base_str += "\nPSF datacube loaded"

        return base_str

    @property
    def _cache_dir(self) -> Path:
        """Directory for all yippy-computed artifacts."""
        d = self.yip_path / "yippy_cache"
        d.mkdir(exist_ok=True)
        return d

    def _source_signature(self) -> str:
        """Short signature over the source YIP FITS files' size and mtime.

        Folded into every cache filename so that an in-place overwrite of the
        source YIP data (same filenames, new content) invalidates cached
        performance curves and PSF datacubes, even when the aperture
        settings, downsample target, and yippy version are unchanged. Cheap:
        stats top-level FITS files only, never reads their contents, and does
        not descend into ``yippy_cache`` itself (``glob`` is non-recursive).
        """
        stat_bits = []
        source_path = self.yip_path
        if getattr(self, "_sinc_delivery", None) is not None:
            source_path = (
                self.yip_path / self._sinc_delivery.manifest["legacy_package_path"]
            )
        for fits_path in sorted(source_path.glob("*.fits")):
            st = fits_path.stat()
            stat_bits.append(f"{fits_path.name}:{st.st_size}:{st.st_mtime_ns}")
        return hashlib.sha1("|".join(stat_bits).encode()).hexdigest()[:8]

    def _cache_identity_base(self, artifact: str, source_roles) -> dict:
        """Identity components shared by every automatically cached artifact.

        Args:
            artifact: Artifact kind, e.g. ``"performance"``.
            source_roles: Keys of ``self._source_files`` whose selected file
                names the artifact depends on.
        """
        identity = {
            "artifact": artifact,
            "schema": _CACHE_SCHEMA_VERSION,
            "sampling": _SAMPLING_CACHE_VERSION,
            "source_signature": self._source_sig,
            "source_files": {role: self._source_files[role] for role in source_roles},
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
        identity = self._cache_identity_base(
            "psf_datacube", ("offax_psf", "offax_offsets")
        )
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
            f"{self._source_sig}_{digest}.npy"
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
        """Complete identity of a single-axis photometry record (schema 4)."""
        from dataclasses import asdict

        identity = self._cache_identity_base(
            "performance",
            ("stellar_intens", "stellar_diams", "offax_psf", "offax_offsets"),
        )
        identity.update(
            {
                "performance_record": 4,
                "aperture_algorithm": "local_component_native_leakage_v1",
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
            f"{self._source_sig}_{digest}.fits"
        )

    def set_aperture_rule(self, rule):
        """Select a static rule, retaining the previously requested records."""
        self.rule = rule
        self._load_or_compute_performance()

    # ------------------------------------------------------------------
    # Performance curve computation (delegates to yippy.performance)
    # ------------------------------------------------------------------

    def _compute_radial_average(self, image, center=None, nbins=None):
        """Compute radial average of a 2D image.

        .. deprecated:: Use :func:`yippy.performance.compute_radial_average`.
        """
        sep_lod, profile = compute_radial_average(
            image, self.pixel_scale_arcsec.value, center=center, nbins=nbins
        )
        # Original returned pixel-unit bin_centers; convert back
        bin_centers_pix = sep_lod / self.pixel_scale_arcsec.value
        return bin_centers_pix, profile

    def _plot_performance_curve(
        self, x, y, title, xlabel, ylabel, marker="o-", log_scale=False, ms=4
    ):
        """Helper method to plot performance curves.

        .. deprecated:: Use :func:`yippy.performance.plot_performance_curve`.
        """
        plot_performance_curve(x, y, title, xlabel, ylabel, marker, log_scale, ms)

    def compute_all_performance_curves(
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
        plot=False,
    ):
        """Build/select one complete record; explicitly imported metadata must match.

        Persistent records store all five quantities on one axis, plus aperture,
        averaging rule, stellar diameter and frame. Corrupt automatic entries
        are cache misses. Explicit imports fail clearly instead of silently
        overriding their physical definitions.
        """
        rule = self.rule if rule is None else rule
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
            f"{self._source_sig}_{expected[:16]}.fits"
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
            record = performance_curves(
                self,
                self.stellar_intens(diameter * lod),
                rule,
                separations_lod=separations_lod,
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
        if plot:
            for name in ("throughput", "core_area_lod2", "raw_contrast"):
                plot_performance_curve(
                    record.separation_lod,
                    getattr(record, name),
                    title=name,
                    xlabel="Separation [lod]",
                    ylabel=name,
                    log_scale=record.interpolation[name] == "log",
                )
        return record

    def occulter_transmission_curve(self, plot=True):
        """Compute occulter transmission curve.

        Delegates to :func:`yippy.performance.compute_occ_trans_curve`.
        """
        sep, occ = compute_occ_trans_curve(self)
        if plot:
            self._plot_performance_curve(
                sep,
                occ,
                title=f"{self.name} Occulter Transmission",
                xlabel="Separation [lam/D]",
                ylabel="Occulter Transmission",
            )
        return sep, occ

    def core_mean_intensity_curve(
        self, stellar_diam_values=None, plot=True, interpolation="keys"
    ):
        """Compute core mean intensity curves.

        Delegates to :func:`yippy.performance.compute_core_mean_intensity_curve`.
        ``interpolation`` selects the spatial reader (``"keys"`` or ``"quintic"``).
        """
        sep, intensities = compute_core_mean_intensity_curve(
            self,
            stellar_diam_values=stellar_diam_values,
            interpolation=interpolation,
        )
        if plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 6))
            for diam, profile in intensities.items():
                plt.plot(sep, profile, "-", label=f"Diam = {diam.value:.1f} lam/D")
            plt.xlabel("Separation [lam/D]")
            plt.ylabel("Core Mean Intensity")
            plt.title(f"{self.name} Core Mean Intensity")
            plt.yscale("log")
            plt.grid(True)
            plt.legend()
            plt.show()
        return sep, intensities

    # ------------------------------------------------------------------
    # Performance metric accessors (use pre-computed interpolators)
    # ------------------------------------------------------------------

    def _convert_separation_to_lod(self, separation):
        """Convert separation value(s) to lambda/D units (scalar or array).

        Args:
            separation (float, Quantity, or array-like):
                The separation value(s), either as scalar(s) in lambda/D or Quantity.

        Returns:
            numpy.ndarray: The separation value(s) in lambda/D.

        Raises:
            ValueError: If the separation has units that are not lambda/D.
        """
        if hasattr(separation, "unit"):
            if separation.unit == lod:
                separation_val = separation.value
            else:
                raise ValueError(
                    f"Separation must be in lambda/D, not {separation.unit}"
                )
        else:
            separation_val = separation

        sep_array = np.atleast_1d(separation_val)

        return sep_array

    def _is_scalar_input(self, sep_values, separation):
        """Check if the original input was scalar."""
        return (len(sep_values) == 1 and np.isscalar(separation)) or (
            hasattr(separation, "shape") and len(separation.shape) == 0
        )

    def throughput(self, separation):
        """Return the throughput at the given separation(s).

        Args:
            separation: Separation(s) in lambda/D (float, Quantity, or array).

        Returns:
            float or numpy.ndarray: The throughput value(s).
        """
        sep_values = self._convert_separation_to_lod(separation)
        result = self.performance.evaluate("throughput", sep_values)
        if self._is_scalar_input(sep_values, separation):
            return float(result[0])
        return result

    def raw_contrast(self, separation):
        """Return the raw contrast at the given separation(s).

        Args:
            separation: Separation(s) in lambda/D (float, Quantity, or array).

        Returns:
            float or numpy.ndarray: The raw contrast value(s).
        """
        sep_values = self._convert_separation_to_lod(separation)
        result = self.performance.evaluate("raw_contrast", sep_values)
        if self.contrast_floor is not None:
            result = np.maximum(result, self.contrast_floor)
        if self._is_scalar_input(sep_values, separation):
            return float(result[0])
        return result

    def noise_floor_exosims(self, separation, contrast_floor=1e-10, ppf=30.0):
        """Return the noise floor in EXOSIMS contrast convention.

        Computes ``max(|raw_contrast|, floor) / ppf``.  The result is in
        contrast-normalized units (per-aperture, divided by throughput).
        EXOSIMS multiplies this by core_thruput to recover C_sr.

        See :doc:`/examples/06_Noise_Floor_Conventions` for details.

        Args:
            separation: Separation(s) in lambda/D.
            contrast_floor (float): Minimum contrast value. Default is 1e-10.
            ppf (float): Post-processing factor. Default is 30.0.

        Returns:
            float or numpy.ndarray: Noise floor in EXOSIMS convention.
        """
        sep_values = self._convert_separation_to_lod(separation)
        raw = self.raw_contrast(separation)
        if np.isscalar(raw):
            raw = np.array([raw])
        contrast = np.maximum(raw, contrast_floor)
        result = contrast / ppf
        if self._is_scalar_input(sep_values, separation):
            return float(result[0])
        return result

    def noise_floor_ayo(self, separation, ppf=30.0, interpolation="keys"):
        """Return the noise floor in AYO/pyEDITH per-pixel convention.

        Computes ``core_mean_intensity(sep) / ppf``.  The result is in
        per-pixel intensity units.  AYO and pyEDITH multiply this by
        ``omega / pixscale**2`` to get the per-aperture noise.

        See :doc:`/examples/06_Noise_Floor_Conventions` for details.

        Args:
            separation: Separation(s) in lambda/D.
            ppf (float): Post-processing factor. Default is 30.0.
            interpolation: Spatial reader, ``"keys"`` or ``"quintic"``.

        Returns:
            float or numpy.ndarray: Noise floor in AYO/pyEDITH convention.
        """
        intensity = self.core_mean_intensity(separation, interpolation=interpolation)
        if np.isscalar(intensity):
            return intensity / ppf
        return np.asarray(intensity) / ppf

    def occulter_transmission(self, separation):
        """Return the occulter transmission at the given separation(s).

        Args:
            separation: Separation(s) in lambda/D.

        Returns:
            float or numpy.ndarray: The occulter transmission value(s).
        """
        sep_values = self._convert_separation_to_lod(separation)
        result = self._occultation_record.evaluate("occulter_transmission", sep_values)
        if self._is_scalar_input(sep_values, separation):
            return float(result[0])
        return result

    def core_area(self, separation):
        """Return the core area at the given separation(s).

        Args:
            separation: Separation(s) in lambda/D.

        Returns:
            float or numpy.ndarray: Core area in (lambda/D)^2.
        """
        sep_values = self._convert_separation_to_lod(separation)
        result = self.performance.evaluate("core_area_lod2", sep_values)
        if self._is_scalar_input(sep_values, separation):
            return float(result[0])
        return result

    def core_mean_intensity(
        self, separation, stellar_diam=0.0 * lod, interpolation="keys"
    ):
        """Return core mean intensity at the given separation(s).

        At a nonzero stellar diameter the profile is the azimuthal mean of
        the stellar intensity map interpolated to that diameter
        (:meth:`stellar_intens`), so the curve and the map describe the same
        star at every diameter, not only at the tabulated ones. The profile is
        computed on first use for each diameter and kept. The default
        diameter (point source), or a package with a single diameter, uses
        the precomputed profile when its spatial reader matches the request.

        Diameters outside the tabulated range return ``NaN``.

        Args:
            separation: Separation(s) in lambda/D.
            stellar_diam: Stellar angular diameter (Quantity in lod
                units, float in lod, or array-like).  Default is
                0.0*lod (point source).
            interpolation: Spatial reader, ``"keys"`` (default) or
                ``"quintic"``. Profiles are cached separately by reader.

        Returns:
            float or numpy.ndarray: Core mean intensity value(s): the
            azimuthal mean, about the optical axis, of the fraction of the
            star's light falling on one pixel. The unit is per pixel of
            ``pixel_scale_arcsec`` (lambda/D per pixel), not per unit solid
            angle; divide by the squared pixel scale for (lambda/D)^-2.
        """
        if interpolation not in ("keys", "quintic"):
            raise ValueError("interpolation must be 'keys' or 'quintic'")
        sep_values = self._convert_separation_to_lod(separation)

        # Extract unitless diameter value
        if hasattr(stellar_diam, "value"):
            diam_val = float(stellar_diam.value)
        else:
            diam_val = float(stellar_diam)

        diams = np.asarray(self.stellar_intens.diams.value)
        stored_reader = getattr(self, "_core_intensity_interpolation", "keys")
        if interpolation != stored_reader:
            if diam_val == 0.0 or len(diams) < 2:
                diam_val = getattr(self, "_core_intensity_diameter", float(diams[0]))
            if not diams.min() <= diam_val <= diams.max():
                result = np.full(np.shape(sep_values), np.nan)
            else:
                result = self._core_intensity_interp_at(
                    diam_val, interpolation=interpolation
                )(sep_values)
        elif diam_val == 0.0 or len(diams) < 2:
            result = self._intensity_record.evaluate("core_mean_intensity", sep_values)
        elif not diams.min() <= diam_val <= diams.max():
            result = np.full(np.shape(sep_values), np.nan)
        else:
            result = self._core_intensity_interp_at(
                diam_val, interpolation=interpolation
            )(sep_values)

        if self._is_scalar_input(sep_values, separation):
            return float(result[0])
        return result

    def _core_intensity_interp_at(self, diam_val: float, interpolation="keys"):
        """Separation interpolator of the core mean intensity at one diameter."""
        key = diam_val if interpolation == "keys" else (diam_val, interpolation)
        interp = self._core_intensity_cache.get(key)
        if interp is None:
            profile = core_mean_intensity_profile(
                self, diam_val * lod, interpolation=interpolation
            )
            interp = self._profile_record(
                self.core_intensity_separations, "core_mean_intensity", profile
            )
            self._core_intensity_cache[key] = interp
        return lambda separation: interp.evaluate("core_mean_intensity", separation)

    # ------------------------------------------------------------------
    # 2D map projections
    # ------------------------------------------------------------------

    def separation_map(self):
        """Pixel-grid separations from the optical axis in lam/D.

        The axis is the one the core mean intensity profile is measured about
        (the shared pixel grid): half a pixel before
        the header's ``XCENTER`` and ``YCENTER``, between the four central
        pixels of an even-sized map. The 2D map projections built on this
        grid therefore agree with their 1D curves.

        Returns:
            numpy.ndarray: (npix, npix) array of separations.
        """
        npix = self.npixels
        cy, cx = self.grid.center_y, self.grid.center_x
        y, x = np.mgrid[:npix, :npix]
        r_pix = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return r_pix * self.pixel_scale_arcsec.value

    def core_mean_intensity_map(self, stellar_diam=0.0 * lod, interpolation="keys"):
        """Azimuthally averaged stellar intensity projected onto the pixel grid.

        Equivalent to computing a radial profile of stellar_intens, fitting a
        1D interpolator, and evaluating it at every pixel's separation. This
        replaces the rotate-and-average approach with a faster, artifact-free
        result.

        See :doc:`/azimuthal_averaging` for the comparison.

        Args:
            stellar_diam: Stellar angular diameter. Default 0.0 lam/D.
            interpolation: Spatial reader, ``"keys"`` or ``"quintic"``.

        Returns:
            numpy.ndarray: (npix, npix) core mean intensity values.
        """
        if self._sinc_delivery is None:
            r = self.separation_map()
        else:
            cy, cx = (
                self.stellar_intens.grid.center_y,
                self.stellar_intens.grid.center_x,
            )
            yy, xx = np.indices(self.stellar_intens.psfs.shape[-2:])
            r = np.hypot(xx - cx, yy - cy) * self.stellar_pixel_scale_arcsec.value
        return np.asarray(
            self.core_mean_intensity(r.ravel(), stellar_diam, interpolation)
        ).reshape(r.shape)

    def noise_floor_ayo_map(
        self, ppf=30.0, stellar_diam=0.0 * lod, interpolation="keys"
    ):
        """Noise floor in AYO/pyEDITH per-pixel convention on the pixel grid.

        Computes ``core_mean_intensity_map / ppf``.

        Args:
            ppf (float): Post-processing factor. Default 30.0.
            stellar_diam: Stellar angular diameter. Default 0.0 lam/D.
            interpolation: Spatial reader, ``"keys"`` or ``"quintic"``.

        Returns:
            numpy.ndarray: (npix, npix) noise floor values.
        """
        return self.core_mean_intensity_map(stellar_diam, interpolation) / ppf

    def throughput_map(self, rules=None):
        """Project selected throughput, or stack requested rules on the last axis."""
        radius = self.separation_map()
        if rules is None:
            return self.performance.evaluate("throughput", radius)
        original = self.performance
        records = [self.compute_all_performance_curves(rule) for rule in rules]
        self.performance, self.rule, self.leakage_rule = (
            original,
            original.rule,
            original.leakage_rule,
        )
        return np.stack([r.evaluate("throughput", radius) for r in records], axis=-1)

    def core_area_map(self, rules=None):
        """Project selected area, or stack requested rules on the last axis."""
        radius = self.separation_map()
        if rules is None:
            return self.performance.evaluate("core_area_lod2", radius)
        original = self.performance
        records = [self.compute_all_performance_curves(rule) for rule in rules]
        self.performance, self.rule, self.leakage_rule = (
            original,
            original.rule,
            original.leakage_rule,
        )
        return np.stack(
            [r.evaluate("core_area_lod2", radius) for r in records], axis=-1
        )

    def throughput_curve(
        self,
        rule=Circular(0.7),  # noqa: B008 - frozen rule
        oversample=CORE_OVERSAMPLE,
        plot=True,
    ):
        """Tabulate the requested rule's throughput."""
        return compute_throughput_curve(self, rule, oversample)

    def raw_contrast_curve(
        self,
        stellar_diam=0 * lod,
        rule=Circular(0.7),  # noqa: B008 - frozen rule
        oversample=CORE_OVERSAMPLE,
        plot=True,
    ):
        """Tabulate the requested rule's raw contrast."""
        return compute_raw_contrast_curve(self, stellar_diam, rule, oversample)

    def core_area_curve(
        self,
        rule=Circular(0.7),  # noqa: B008 - frozen rule
        fit_gaussian=False,
        oversample=CORE_OVERSAMPLE,
        plot=True,
    ):
        """Tabulate the requested rule's area."""
        return compute_core_area_curve(self, rule, fit_gaussian, oversample)

    # ------------------------------------------------------------------
    # Export (delegates to yippy.export)
    # ------------------------------------------------------------------

    def to_exosims(self, units="LAMBDA/D", *, separations_lod=None):
        """Write the selected record through the common evaluator."""
        from .export import export_exosims

        return export_exosims(self, units=units, separations_lod=separations_lod)

    def dump_ayo_csv(
        self,
        output_path,
        sep_min=0.125,
        sep_max=32.0,
        sep_step=0.25,
        contrast_floor=None,
        ppf=30.0,
        *,
        separations_lod=None,
    ):
        """Export performance curves in AYO-compatible CSV format.

        Delegates to :func:`yippy.export.export_ayo_csv`.
        """
        return export_ayo_csv(
            self,
            output_path,
            sep_min=sep_min,
            sep_max=sep_max,
            sep_step=sep_step,
            contrast_floor=contrast_floor,
            ppf=ppf,
            separations_lod=separations_lod,
        )
