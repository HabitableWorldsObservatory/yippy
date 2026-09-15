"""Base coronagraph class."""

import hashlib
from pathlib import Path

import astropy.io.fits as pyfits
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from lod_unit import lod
from tqdm import tqdm

from ._precision import dtype_tag, float_dtype
from ._version import __version__
from .export import export_ayo_csv
from .header import HeaderData
from .logger import logger
from .offjax import OffJAX
from .performance import (
    compute_all_performance_curves as _compute_all_perf,
)
from .performance import (
    compute_core_area_curve,
    compute_core_mean_intensity_curve,
    compute_occ_trans_curve,
    compute_radial_average,
    compute_raw_contrast_curve,
    compute_throughput_curve,
    plot_performance_curve,
)
from .sky_trans import SkyTrans
from .stellar_intens import StellarIntens
from .util import cache_identity_digest, read_performance_cache_key

# Invalidate artifacts generated before conservative PSF rebinning. Unlike a
# release version this also protects development checkouts using the same tag.
_SAMPLING_CACHE_VERSION = 1

# Namespace of automatically managed cache entries. Schema 2 keys every entry
# by a digest of its complete calculation identity; entries from earlier
# schemas have different names and are never read.
_CACHE_SCHEMA_VERSION = 2

# Oversampling factor for automatically computed performance tables.
_DEFAULT_OVERSAMPLE = 2


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
        aperture_radius_lod: float = 0.7,
        contrast_floor: float | None = None,
        use_inscribed_diameter: bool = False,
        psf_trunc_ratio: float | None = None,
        interp_order: int = 1,
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
            aperture_radius_lod (float):
                Aperture radius in lambda/D for throughput and contrast calculations.
                Default is 0.7 (AYO typically uses 0.85).
            contrast_floor (float | None):
                Minimum contrast value for engineering stability floor.
                If provided, raw contrast values are floored at this value.
                Typical AYO value is 1e-10. Default is None (no floor).
            use_inscribed_diameter (bool):
                Whether to use the inscribed diameter for lam/D calculations.
                When True, input separations are scaled by D/D_INSC to
                convert from inscribed to circumscribed units. Default False.
            psf_trunc_ratio (float | None):
                PSF truncation ratio for throughput and core area computation.
                When set, pixels above ``ratio * peak`` define the aperture
                (matching AYO's ``photap_frac`` / ``omega_lod``).
                When None, a fixed circular aperture is used instead.
                Typical AYO value is 0.3. Default is None.
            interp_order (int):
                B-spline order for performance-curve interpolators.
                Use 1 for linear (default) or 3 for cubic.
        """
        ###################
        # Read input data #
        ###################
        yip_path = Path(yip_path)
        self.yip_path = yip_path

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
        self.aperture_radius_lod = aperture_radius_lod
        self.contrast_floor = contrast_floor
        self.psf_trunc_ratio = psf_trunc_ratio

        # Get header and calculate the lambda/D value
        stellar_intens_header = pyfits.getheader(Path(yip_path, stellar_intens_file), 0)

        # Get pixel scale with units
        self.header = HeaderData.from_fits_header(stellar_intens_header)
        self.pixel_scale_arcsec = self.header.pixscale
        self.frac_obscured = self.header.obscured

        # Optionally use inscribed diameter for lam/D calculations (AYO compatibility)
        # When enabled, input separations are scaled by D/D_INSC before querying
        self.use_inscribed_diameter = use_inscribed_diameter
        self._diameter_ratio = 1.0  # Default: no scaling
        if use_inscribed_diameter:
            if (
                self.header.diameter_inscribed is not None
                and self.header.diameter is not None
            ):
                self._diameter_ratio = float(
                    self.header.diameter / self.header.diameter_inscribed
                )
                logger.info(
                    f"Using inscribed diameter: separations scaled by "
                    f"{self._diameter_ratio:.4f}"
                )
            else:
                logger.warning(
                    "use_inscribed_diameter=True but D_INSC not found in header"
                )

        # Stellar intensity of the star being observed as function of stellar
        # angular diameter (unitless)
        self.stellar_intens = StellarIntens(
            yip_path, stellar_intens_file, stellar_diam_file
        )

        # Offaxis PSF of the planet as function of separation from the star
        self.offax = OffJAX(
            yip_path,
            offax_data_file,
            offax_offsets_file,
            self.pixel_scale_arcsec,
            x_symmetric,
            y_symmetric,
            downsample_shape=downsample_shape,
        )

        # Update pixel_scale_arcsec if downsampling was applied
        if downsample_shape is not None:
            self.pixel_scale_arcsec = self.offax.pixel_scale_arcsec

        # Get the sky_trans mask
        self.sky_trans = SkyTrans(yip_path, sky_trans_file)

        # PSF datacube here is a 4D array of PSFs at each pixel (x psf offset,
        # y psf offset, x, y). Given the computational cost of generating this
        # datacube, it is only generated when needed.
        self.has_psf_datacube = False
        self.use_quarter_psf_datacube = use_quarter_psf_datacube

        # Shape of the images in the PSFs
        # Use actual PSF shape from offax (accounts for downsampling)
        self.psf_shape = np.array(self.offax.psf_shape)
        assert self.psf_shape[0] == self.psf_shape[1], "PSF must be square"
        self.npixels = self.psf_shape[0]

        self.interp_order = interp_order

        self._load_or_compute_performance()

        logger.info(f"Created {yip_path.stem}")

    def _load_or_compute_performance(self):
        """Build performance interpolators from a valid cache entry or afresh.

        Shared by the constructor and ``set_psf_trunc_ratio`` so both use the
        same settings and the same stored-value meaning. An entry is reused
        only when its recorded identity digest equals the identity of the
        current request; anything else (missing, legacy, unreadable, or
        written under other settings) is a miss and is overwritten.
        """
        perf_file = self._perf_filename()
        perf_path = self._perf_dir / perf_file
        expected = cache_identity_digest(self._auto_perf_identity())
        hit = perf_path.exists() and read_performance_cache_key(perf_path) == expected

        if hit:
            logger.info(f"Loading performance metrics from {perf_path}")
        else:
            if perf_path.exists():
                logger.info(f"Ignoring incompatible performance cache {perf_path}")
            if self.psf_trunc_ratio is not None:
                logger.info(
                    f"Computing performance with PSF trunc ratio "
                    f"= {self.psf_trunc_ratio}..."
                )
            else:
                logger.info("Computing all performance metrics...")
        self.compute_all_performance_curves(
            aperture_radius_lod=self.aperture_radius_lod,
            oversample=_DEFAULT_OVERSAMPLE,
            save_to_fits=not hit,
            performance_file=perf_file,
            load_from_file=perf_file if hit else None,
            cache_dir=self._perf_dir,
            plot=False,
            psf_trunc_ratio=self.psf_trunc_ratio,
            interp_order=self.interp_order,
        )

    def create_psf_datacube(self, batch_size=128):
        """Load the PSF datacube from a file or generate it if it doesn't exist.

        The PSF datacube is a 4D array of PSFs at each pixel (x psf offset,
        y psf offset, x, y). Given the computational cost of generating this
        datacube, it is only generated when needed and saved to a numpy binary
        file in the yip_path directory.

        Args:
            batch_size (int):
                Number of PSFs to generate in each batch. Default is 128.
        """
        backend = jax.default_backend().lower()
        if backend in ("gpu", "tpu"):
            return self._create_psf_datacube_gpu(batch_size=batch_size)
        ext = "_quarter" if self.use_quarter_psf_datacube else ""
        datacube_path = self._datacube_cache_path
        if datacube_path.exists():
            logger.info(f"Loading PSF datacube from {datacube_path}.")
            psfs = jnp.asarray(jnp.load(datacube_path))
        else:
            # Create data cube of spatially dependent PSFs.
            psfs_shape = (*self.psf_shape, *self.psf_shape)
            psfs = np.zeros(psfs_shape, dtype=float_dtype())
            if not self.use_quarter_psf_datacube:
                pixel_lod = (
                    (np.arange(self.npixels) - ((self.npixels - 1) // 2))
                    * u.pixel
                    * self.pixel_scale_arcsec
                ).value
            else:
                center_idx = (self.npixels - 1) // 2
                pixel_lod = (
                    (np.arange(center_idx, self.npixels) - center_idx)
                    * u.pixel
                    * self.pixel_scale_arcsec
                ).value
            n_src = len(pixel_lod)
            psfs_shape = (n_src, n_src, *self.psf_shape)
            psfs = np.zeros(psfs_shape, dtype=float_dtype())

            # Get the pixel coordinates for the PSF evaluations
            x_lod, y_lod = np.meshgrid(pixel_lod, pixel_lod, indexing="xy")
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
                    ] = batch_psfs
                    pb.update(batch_points.shape[0])
            jnp.save(datacube_path, psfs)
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
        datacube_path = self._datacube_cache_path
        if datacube_path.exists():
            logger.info(f"Loading PSF datacube from {datacube_path}.")
            psfs = jnp.asarray(jnp.load(datacube_path))
        else:
            if not self.use_quarter_psf_datacube:
                pixel_lod = (
                    (np.arange(self.npixels) - ((self.npixels - 1) // 2))
                    * u.pixel
                    * self.pixel_scale_arcsec
                ).value
            else:
                center_idx = (self.npixels - 1) // 2
                pixel_lod = (
                    (np.arange(center_idx, self.npixels) - center_idx)
                    * u.pixel
                    * self.pixel_scale_arcsec
                ).value

            n_src = len(pixel_lod)
            psfs_shape = (n_src, n_src, *self.psf_shape)

            x_lod, y_lod = np.meshgrid(pixel_lod, pixel_lod, indexing="xy")
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
                    flat_view[i : i + batch_size] = np.asarray(batch_psfs)
                    pb.update(batch_x.shape[0])

            np.save(datacube_path, psfs)
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

        # Add information about performance metrics
        interp_info = []
        if hasattr(self, "throughput_interp"):
            interp_info.append("throughput")
        if hasattr(self, "raw_contrast_interp"):
            interp_info.append("raw contrast")
        if hasattr(self, "occ_trans_interp"):
            interp_info.append("occulter transmission")
        if hasattr(self, "core_area_interp"):
            interp_info.append("core area")
        if hasattr(self, "core_intensity_interp"):
            interp_info.append("core mean intensity")

        if interp_info:
            base_str += f"\nInterpolators available: {', '.join(interp_info)}"

        if hasattr(self, "IWA"):
            base_str += f"\nInner Working Angle: {self.IWA:.2f}"

        if hasattr(self, "OWA"):
            base_str += f"\nOuter Working Angle: {self.OWA:.2f}"

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
        for fits_path in sorted(self.yip_path.glob("*.fits")):
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
        return {
            "artifact": artifact,
            "schema": _CACHE_SCHEMA_VERSION,
            "sampling": _SAMPLING_CACHE_VERSION,
            "source_signature": self._source_sig,
            "source_files": {role: self._source_files[role] for role in source_roles},
            "npixels": int(self.npixels),
            "dtype": dtype_tag(),
        }

    def _datacube_identity(self) -> dict:
        """Complete identity of the PSF datacube this object would store.

        The physical symmetry flags only change PSFs synthesized at negative
        offsets, which a quarter cube never evaluates, so they are part of the
        identity of full cubes only.
        """
        identity = self._cache_identity_base(
            "psf_datacube", ("offax_psf", "offax_offsets")
        )
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
        self, aperture_radius_lod, psf_trunc_ratio, oversample, stellar_diam
    ) -> dict:
        """Complete identity of a stored performance table.

        The table holds separation, throughput and unfloored raw contrast, so
        the identity covers everything those depend on: the source files and
        their generation, realized resolution and dtype, the circular aperture
        radius (which sets contrast even in truncation mode), the aperture
        mode and threshold, the effective oversampling, and the stellar
        diameter of the contrast calculation. Settings applied after loading
        (contrast floor, inscribed-diameter scaling, interpolation order) and
        quantities recomputed on every construction (core area, occulter
        transmission, core mean intensity) are not part of it.
        """
        identity = self._cache_identity_base(
            "performance",
            ("stellar_intens", "stellar_diams", "offax_psf", "offax_offsets"),
        )
        identity.update(
            {
                "yippy_version": __version__,
                "aperture": "circular" if psf_trunc_ratio is None else "truncation",
                "aperture_radius_lod": aperture_radius_lod,
                "psf_trunc_ratio": psf_trunc_ratio,
                "oversample": oversample,
                "stellar_diam_lod": getattr(stellar_diam, "value", stellar_diam),
            }
        )
        return identity

    def _auto_perf_identity(self) -> dict:
        """Identity of the automatically managed table for current settings."""
        return self._perf_identity(
            aperture_radius_lod=self.aperture_radius_lod,
            psf_trunc_ratio=self.psf_trunc_ratio,
            oversample=_DEFAULT_OVERSAMPLE,
            stellar_diam=self.stellar_intens.diams[0],
        )

    def _perf_filename(self) -> str:
        """Performance cache filename for the current settings.

        The readable prefix gives the aperture mode, realized pixel count,
        dtype and source signature; the trailing digest identifies the
        complete calculation (see ``_perf_identity``), so parameters that
        differ in any bit never share a name. The file's header carries the
        full digest, which is checked before reuse.
        """
        mode = "aper" if self.psf_trunc_ratio is None else "trunc"
        digest = cache_identity_digest(self._auto_perf_identity())[:16]
        return (
            f"perf_{mode}_{self.npixels}px_{dtype_tag()}_{self._source_sig}_"
            f"{digest}.fits"
        )

    def set_psf_trunc_ratio(self, ratio: float) -> None:
        """Switch PSF truncation ratio and rebuild the performance curves.

        Uses the same settings, cache identity and stored-value definitions as
        construction with ``psf_trunc_ratio=ratio``, so the resulting curves do
        not depend on whether a table was produced here or by the constructor.
        Results are cached to ``yippy_cache/performance/`` for reuse.

        Args:
            ratio: PSF truncation ratio (e.g. 0.3).
        """
        self.psf_trunc_ratio = ratio
        self._load_or_compute_performance()

        logger.info(
            f"Switched to trunc_ratio={ratio:.2f} "
            f"(IWA={self.IWA:.2f}, OWA={self.OWA:.2f})"
        )

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
        aperture_radius_lod=0.7,
        stellar_diam=None,
        fit_gaussian_for_core_area=False,
        use_phot_aperture_as_min=False,
        oversample=2,
        save_to_fits=True,
        performance_file="coro_perf.fits",
        load_from_file=None,
        cache_dir=None,
        plot=False,
        psf_trunc_ratio=None,
        interp_order=1,
    ):
        """Compute all coronagraph performance curves at once.

        Delegates to :func:`yippy.performance.compute_all_performance_curves`.
        """
        return _compute_all_perf(
            self,
            aperture_radius_lod=aperture_radius_lod,
            stellar_diam=stellar_diam,
            fit_gaussian_for_core_area=fit_gaussian_for_core_area,
            use_phot_aperture_as_min=use_phot_aperture_as_min,
            oversample=oversample,
            save_to_fits=save_to_fits,
            performance_file=performance_file,
            load_from_file=load_from_file,
            cache_dir=cache_dir,
            plot=plot,
            psf_trunc_ratio=psf_trunc_ratio,
            interp_order=interp_order,
        )

    def _compute_performance_metrics(
        self,
        stellar_diam=0.0 * lod,
        aperture_radius_lod=0.7,
        fit_gaussian_for_core_area=False,
        use_phot_aperture_as_min=False,
        oversample=2,
        compute_throughput=True,
        compute_contrast=True,
        compute_core_area=True,
    ):
        """Compute performance metrics.

        .. deprecated:: Use individual functions in :mod:`yippy.performance`.
        """
        result = {}
        if compute_throughput:
            sep, vals = compute_throughput_curve(
                self,
                aperture_radius_lod=aperture_radius_lod,
                oversample=oversample,
            )
            result["separations"] = sep
            result["throughput"] = vals
        if compute_contrast:
            sep_c, vals_c = compute_raw_contrast_curve(
                self,
                stellar_diam=stellar_diam,
                aperture_radius_lod=aperture_radius_lod,
                oversample=oversample,
            )
            result.setdefault("separations", sep_c)
            result["raw_contrast"] = vals_c
        if compute_core_area:
            sep_a, vals_a = compute_core_area_curve(
                self,
                aperture_radius_lod=aperture_radius_lod,
                fit_gaussian=fit_gaussian_for_core_area,
                use_phot_aperture_as_min=use_phot_aperture_as_min,
                oversample=oversample,
            )
            result.setdefault("separations", sep_a)
            result["core_area"] = vals_a
        return result

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

    def core_mean_intensity_curve(self, stellar_diam_values=None, plot=True):
        """Compute core mean intensity curves.

        Delegates to :func:`yippy.performance.compute_core_mean_intensity_curve`.
        """
        sep, intensities = compute_core_mean_intensity_curve(
            self,
            stellar_diam_values=stellar_diam_values,
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

        if self._diameter_ratio != 1.0:
            sep_array = sep_array * self._diameter_ratio

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
        result = self.throughput_interp(sep_values)
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
        log_result = self._log_contrast_interp(sep_values)
        result = np.power(10.0, log_result)
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

    def noise_floor_ayo(self, separation, ppf=30.0):
        """Return the noise floor in AYO/pyEDITH per-pixel convention.

        Computes ``core_mean_intensity(sep) / ppf``.  The result is in
        per-pixel intensity units.  AYO and pyEDITH multiply this by
        ``omega / pixscale**2`` to get the per-aperture noise.

        See :doc:`/examples/06_Noise_Floor_Conventions` for details.

        Args:
            separation: Separation(s) in lambda/D.
            ppf (float): Post-processing factor. Default is 30.0.

        Returns:
            float or numpy.ndarray: Noise floor in AYO/pyEDITH convention.
        """
        intensity = self.core_mean_intensity(separation)
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
        result = self.occ_trans_interp(sep_values)
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
        result = self.core_area_interp(sep_values)
        if self._is_scalar_input(sep_values, separation):
            return float(result[0])
        return result

    def core_mean_intensity(self, separation, stellar_diam=0.0 * lod):
        """Return core mean intensity at the given separation(s).

        Interpolates over both separation and stellar angular diameter
        using a 2D ``RegularGridInterpolator`` when multiple diameters
        are available and a non-default diameter is requested.  Uses
        the faster 1D spline for the default diameter (point source)
        or when the yield input package contains only one diameter.

        Out-of-bounds queries on the 2D grid return ``NaN``.

        Args:
            separation: Separation(s) in lambda/D.
            stellar_diam: Stellar angular diameter (Quantity in lod
                units, float in lod, or array-like).  Default is
                0.0*lod (point source).

        Returns:
            float or numpy.ndarray: Core mean intensity value(s).
        """
        sep_values = self._convert_separation_to_lod(separation)

        # Extract unitless diameter value
        if hasattr(stellar_diam, "value"):
            diam_val = float(stellar_diam.value)
        else:
            diam_val = float(stellar_diam)

        # Use 1D spline for the default diameter (fast path, extrapolates
        # naturally).  Route through 2D only for non-default diameters
        # where cross-diameter interpolation is needed.
        if diam_val != 0.0 and self.core_intensity_interp_2d is not None:
            points = np.column_stack(
                [
                    sep_values,
                    np.full_like(sep_values, diam_val),
                ]
            )
            result = self.core_intensity_interp_2d(points)
        else:
            result = self.core_intensity_interp(sep_values)

        if self._is_scalar_input(sep_values, separation):
            return float(result[0])
        return result

    # ------------------------------------------------------------------
    # 2D map projections
    # ------------------------------------------------------------------

    def separation_map(self):
        """Pixel-grid separations from the coronagraph center in lam/D.

        Returns:
            numpy.ndarray: (npix, npix) array of separations.
        """
        npix = self.npixels
        cx = self.header.xcenter
        cy = self.header.ycenter
        y, x = np.mgrid[:npix, :npix]
        r_pix = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return r_pix * self.pixel_scale_arcsec.value

    def core_mean_intensity_map(self, stellar_diam=0.0 * lod):
        """Azimuthally averaged stellar intensity projected onto the pixel grid.

        Equivalent to computing a radial profile of stellar_intens, fitting a
        1D interpolator, and evaluating it at every pixel's separation. This
        replaces the rotate-and-average approach with a faster, artifact-free
        result.

        See :doc:`/azimuthal_averaging` for the comparison.

        Args:
            stellar_diam: Stellar angular diameter. Default 0.0 lam/D.

        Returns:
            numpy.ndarray: (npix, npix) core mean intensity values.
        """
        r = self.separation_map()
        return np.asarray(self.core_mean_intensity(r.ravel(), stellar_diam)).reshape(
            r.shape
        )

    def noise_floor_ayo_map(self, ppf=30.0, stellar_diam=0.0 * lod):
        """Noise floor in AYO/pyEDITH per-pixel convention on the pixel grid.

        Computes ``core_mean_intensity_map / ppf``.

        Args:
            ppf (float): Post-processing factor. Default 30.0.
            stellar_diam: Stellar angular diameter. Default 0.0 lam/D.

        Returns:
            numpy.ndarray: (npix, npix) noise floor values.
        """
        return self.core_mean_intensity_map(stellar_diam) / ppf

    def throughput_map(self, psf_trunc_ratios=None):
        """Throughput projected from the 1D curve onto the pixel grid.

        Args:
            psf_trunc_ratios (array-like or None):
                If provided, returns a stacked (npix, npix, nratios) array
                with one slice per truncation ratio. Each ratio triggers a
                fresh performance computation. If None, uses the current
                throughput interpolator.

        Returns:
            numpy.ndarray: (npix, npix) or (npix, npix, nratios) throughput.
        """
        r = self.separation_map()
        r_flat = r.ravel()
        if psf_trunc_ratios is None:
            return np.asarray(self.throughput_interp(r_flat)).reshape(r.shape)
        original_ratio = self.psf_trunc_ratio
        maps = []
        for ratio in psf_trunc_ratios:
            self.set_psf_trunc_ratio(ratio)
            maps.append(np.asarray(self.throughput_interp(r_flat)).reshape(r.shape))
        if original_ratio is not None:
            self.set_psf_trunc_ratio(original_ratio)
        return np.stack(maps, axis=-1)

    def core_area_map(self, psf_trunc_ratios=None):
        """Core area (omega) projected from the 1D curve onto the pixel grid.

        Args:
            psf_trunc_ratios (array-like or None):
                If provided, returns a stacked (npix, npix, nratios) array.
                If None, uses the current core_area interpolator.

        Returns:
            numpy.ndarray: (npix, npix) or (npix, npix, nratios) core area
                in (lam/D)^2.
        """
        r = self.separation_map()
        r_flat = r.ravel()
        if psf_trunc_ratios is None:
            return np.asarray(self.core_area_interp(r_flat)).reshape(r.shape)
        original_ratio = self.psf_trunc_ratio
        maps = []
        for ratio in psf_trunc_ratios:
            self.set_psf_trunc_ratio(ratio)
            maps.append(np.asarray(self.core_area_interp(r_flat)).reshape(r.shape))
        if original_ratio is not None:
            self.set_psf_trunc_ratio(original_ratio)
        return np.stack(maps, axis=-1)

    # ------------------------------------------------------------------
    # Standalone curve methods (for plotting individual curves)
    # ------------------------------------------------------------------

    def throughput_curve(self, aperture_radius_lod=0.7, oversample=1, plot=True):
        """Compute and optionally plot the throughput curve.

        Delegates to :func:`yippy.performance.compute_throughput_curve`.
        """
        sep, vals = compute_throughput_curve(
            self,
            aperture_radius_lod=aperture_radius_lod,
            oversample=oversample,
        )
        if plot:
            self._plot_performance_curve(
                sep,
                vals,
                title=f"{self.name} Throughput",
                xlabel="Separation [lam/D]",
                ylabel="Throughput",
                ms=6,
            )
        return sep, vals

    def raw_contrast_curve(
        self, stellar_diam=0 * lod, aperture_radius_lod=0.7, oversample=2, plot=True
    ):
        """Compute and optionally plot the raw contrast curve.

        Delegates to :func:`yippy.performance.compute_raw_contrast_curve`.
        """
        sep, vals = compute_raw_contrast_curve(
            self,
            stellar_diam=stellar_diam,
            aperture_radius_lod=aperture_radius_lod,
            oversample=oversample,
        )
        if plot:
            self._plot_performance_curve(
                sep,
                vals,
                title=f"{self.name} Raw Contrast",
                xlabel="Separation [lam/D]",
                ylabel="Raw Contrast",
                log_scale=True,
            )
        return sep, vals

    def core_area_curve(
        self,
        aperture_radius_lod=0.7,
        fit_gaussian=False,
        use_phot_aperture_as_min=False,
        oversample=2,
        plot=True,
    ):
        """Compute and optionally plot the core area curve.

        Delegates to :func:`yippy.performance.compute_core_area_curve`.
        """
        sep, vals = compute_core_area_curve(
            self,
            aperture_radius_lod=aperture_radius_lod,
            fit_gaussian=fit_gaussian,
            use_phot_aperture_as_min=use_phot_aperture_as_min,
            oversample=oversample,
        )
        if plot:
            suffix = " (Gaussian fit)" if fit_gaussian else " (fixed aperture)"
            self._plot_performance_curve(
                sep,
                vals,
                title=f"{self.name} Core Area{suffix}",
                xlabel="Separation [lam/D]",
                ylabel="Core Area [(lam/D)**2]",
            )
        return sep, vals

    # ------------------------------------------------------------------
    # Export (delegates to yippy.export)
    # ------------------------------------------------------------------

    def to_exosims(
        self,
        aperture_radius_lod=0.7,
        fit_gaussian_for_core_area=False,
        use_phot_aperture_as_min=False,
        units="LAMBDA/D",
    ):
        """Save performance curves in EXOSIMS format.

        Delegates to :func:`yippy.export.export_exosims`.
        """
        from .export import export_exosims

        return export_exosims(
            self,
            aperture_radius_lod=aperture_radius_lod,
            fit_gaussian_for_core_area=fit_gaussian_for_core_area,
            use_phot_aperture_as_min=use_phot_aperture_as_min,
            units=units,
        )

    def dump_ayo_csv(
        self,
        output_path,
        sep_min=0.125,
        sep_max=32.0,
        sep_step=0.25,
        contrast_floor=1e-10,
        ppf=30.0,
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
        )
