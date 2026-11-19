"""A toy Lyot coronagraph for the yippy.sinc documentation.

The model is deliberately small: a circular entrance pupil of width D, a hard
occulting disk in the first focal plane, a circular Lyot stop and a fixed
detector, connected by matrix Fourier transforms. Every operation after the
source tilt is linear and independent of the source position, which is the
property the documentation pages rely on. It is a teaching model, not a
design: use a physical-optics package for real coronagraphs.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from optixstuff.grids import Frame, Measure, PixelGrid

import yippy.sinc as sc

jax.config.update("jax_enable_x64", True)


def _axis(count, step):
    """Return centered sample positions with the given step."""
    return (jnp.arange(count) - (count - 1) / 2) * step


@dataclass(frozen=True)
class ToyCoronagraph:
    """Planet intensity images of a toy Lyot coronagraph.

    Attributes:
        n_pupil: Pupil samples across the entrance diameter D.
        occulter_lod: Occulting disk radius in lambda/D.
        lyot: Lyot stop diameter as a fraction of D.
        n_pix: Detector pixels per side.
        pixscale_lod: Detector pixel pitch in lambda/D.
        pupil_shift: Decenter (dx, dy) of the entrance pupil, in units of D.
            A shift along one axis keeps the intensity mirror symmetric (see
            the quadrant-symmetry page); a diagonal shift breaks it.
    """

    n_pupil: int = 48
    occulter_lod: float = 2.5
    lyot: float = 0.8
    n_pix: int = 96
    pixscale_lod: float = 0.25
    pupil_shift: tuple[float, float] = (0.0, 0.0)

    @property
    def pupil_coords(self):
        """Pupil sample coordinates (u, v) in units of D."""
        u = _axis(self.n_pupil, 1.0 / self.n_pupil)
        return jnp.meshgrid(u, u)

    @property
    def pupil(self):
        """Entrance pupil transmission, a disk of diameter D."""
        uu, vv = self.pupil_coords
        dx, dy = self.pupil_shift
        return ((uu - dx) ** 2 + (vv - dy) ** 2 <= 0.25).astype(jnp.float64)

    @property
    def detector_x(self):
        """Detector pixel-center coordinates along one axis, in lambda/D."""
        return _axis(self.n_pix, self.pixscale_lod)

    def phase_ramp(self, source_xy):
        """Return the pupil phase, in radians, that a source tilt imposes."""
        uu, vv = self.pupil_coords
        return 2 * jnp.pi * (uu * source_xy[0] + vv * source_xy[1])

    def field(self, source_xy):
        """Complex detector field for one source position in lambda/D."""
        uu, vv = self.pupil_coords
        u = uu[0]
        stop = (uu**2 + vv**2 <= (self.lyot / 2) ** 2).astype(jnp.float64)
        n_occ = 40
        xi = _axis(n_occ, 2 * self.occulter_lod / n_occ)
        xx, yy = jnp.meshgrid(xi, xi)
        occulter = (xx**2 + yy**2 <= self.occulter_lod**2).astype(jnp.float64)
        du = 1.0 / self.n_pupil
        to_focal = jnp.exp(-2j * jnp.pi * jnp.outer(xi, u)) * du
        to_lyot = jnp.exp(2j * jnp.pi * jnp.outer(u, xi)) * (xi[1] - xi[0])
        to_detector = jnp.exp(-2j * jnp.pi * jnp.outer(self.detector_x, u)) * du
        tilted = self.pupil * jnp.exp(1j * self.phase_ramp(source_xy))
        focal = to_focal @ tilted @ to_focal.T
        lyot_field = tilted - to_lyot @ (occulter * focal) @ to_lyot.T
        return to_detector @ (stop * lyot_field) @ to_detector.T

    def psf(self, source_xy):
        """Planet intensity image(s) for source positions [..., 2] in lambda/D.

        Images are point samples of intensity at pixel centers, normalized
        by the peak of the unocculted, unstopped image of the full pupil.
        """
        source_xy = jnp.asarray(source_xy, dtype=jnp.float64)
        peak = (jnp.sum(self.pupil) / self.n_pupil**2) ** 2

        def one(source):
            return jnp.abs(self.field(source)) ** 2 / peak

        flat = source_xy.reshape(-1, 2)
        images = jax.jit(jax.vmap(one))(flat)
        return images.reshape(*source_xy.shape[:-1], self.n_pix, self.n_pix)


def products_for(toy, *, identity="toy-lyot"):
    """Return the yippy.sinc product declaration for a toy coronagraph.

    The declarations state what is true of the toy: a monochromatic product,
    point samples of intensity, an entrance support of width D in both axes,
    and fixed linear optics after the source tilt.
    """
    center = (toy.n_pix - 1) / 2
    pixels = PixelGrid(
        (toy.n_pix, toy.n_pix),
        (jnp.array([center, center]))[0],
        (jnp.array([center, center]))[1],
        jnp.eye(2) * toy.pixscale_lod,
        "lod",
        Frame(500.0, 6.0, "circumscribed"),
    )
    return sc.Products(
        pixels=pixels,
        normalization=sc.Normalization(
            "incident_photon",
            jnp.array([1.0]),
            jnp.array([jnp.nan]),
            (),
            "toy-unocculted-peak",
        ),
        kind="monochromatic",
        wavelength_nm=jnp.array([500.0]),
        limits_nm=jnp.array([[500.0, 500.0]]),
        sampling_min_nm=jnp.array([500.0]),
        campaign_sampling_min_nm=jnp.array(jnp.nan),
        support_extent_xy=jnp.ones((1, 2)),
        support_evidence=jnp.array([1], dtype=jnp.int32),
        support_source_ids=("toy-circular-pupil",),
        premises=sc.Premises(
            jnp.ones(6, dtype=bool),
            jnp.ones(6, dtype=jnp.int32),
            ("toy-linear-optics",) * 6,
        ),
        optical_id="toy",
        spectral_weight_id="mono",
        identity=identity,
        measure=Measure.density("grid"),
    )


def quadrant_delivery(toy=None, *, spacing=0.38, nodes=31, domain_half=5.0):
    """Build the keyword arguments of ``yippy.sinc.admit_grid`` for a quadrant grid.

    Args:
        toy: The coronagraph to propagate; the default toy if omitted.
        spacing: Node spacing in lambda/D, the same on both axes.
        nodes: Nodes per axis, indices 0 through nodes - 1.
        domain_half: Half-width of the square served box, in lambda/D.

    Returns:
        A dict ready for ``yippy.sinc.admit_grid(**delivery)``.
    """
    toy = ToyCoronagraph() if toy is None else toy
    lattice = sc.source_grid(
        jnp.zeros(2),
        jnp.full(2, spacing),
        (nodes, nodes),
        Frame(500.0, 6.0, "circumscribed"),
    )
    iy, ix = jnp.indices(lattice.shape, dtype=jnp.int32)
    indices = jnp.stack((ix.ravel(), iy.ravel()), axis=1)
    offsets = indices * spacing
    signed = jnp.array([[-3, 2], [3, -2], [-3, -2]], dtype=jnp.int32)
    mirrors = sc.MirrorCheckSet(
        signed * spacing,
        signed,
        toy.psf(signed * spacing),
        jnp.zeros(3, dtype=jnp.int32),
        "toy-direct-propagation",
    )
    return dict(
        intensity_nodes=toy.psf(offsets)[None],
        offsets_xy_lod=offsets,
        declared_lattice=lattice,
        indices_xy=indices,
        products=products_for(toy),
        reflection=sc.Reflection(
            "two_axis_pixel_flip_v1",
            jnp.array([True, True]),
            "toy-symmetry-declaration",
            "circular pupil, occulter and stop",
            jnp.array(1, dtype=jnp.int32),
            "toy-mirror-checks",
            jnp.array([False]),
        ),
        domain=sc.Domain(
            "box",
            "closed",
            jnp.array([[-domain_half, -domain_half], [domain_half, domain_half]]),
            jnp.zeros(2),
        ),
        memory=sc.MemoryBudget(2_000_000_000, 400_000_000, 400_000_000, 4_000_000_000),
        mirror_checks=mirrors,
    )


def reader_for(delivery):
    """Admit a delivery and return an ``Interpolator`` over its domain."""
    grid = sc.admit_grid(**delivery).require()
    return sc.Interpolator(
        grid,
        delivery["domain"],
        sc.KernelProfile(),
        sc.ReadPolicy(),
        sc.Tiling(pixel_tile=1024),
        delivery["memory"],
    )


def kernel_error(rho, kernel=None, trials=40):
    """Typical reconstruction error of a 32-tap kernel on random signals.

    This is the largest error over 40 random draws, each a sum of twelve
    cosines, so it is a typical-signal measurement and not a worst case: use
    ``kernel_error_worst`` for the worst case over every single tone. The signals
    are random band-limited sums (twelve cosines with frequencies up to
    ``rho / 2`` cycles per sample), sampled at the integers and rebuilt at
    random offsets inside one cell. This is the dashed typical-signal curve
    on the kernel page.

    Args:
        rho: Oversampling factor; Nyquist is 1.
        kernel: Function of the offset ``q - tap`` in samples; the yippy.sinc
            Kaiser-sinc weights when omitted.
        trials: Number of random signals.

    Returns:
        The largest absolute error divided by the signal's total amplitude.
    """
    import numpy as np

    from yippy.sinc.kernel import weights

    if kernel is None:

        def kernel(t):
            return np.asarray(weights(jnp.asarray(t)))

    worst = 0.0
    for seed in range(trials):
        draw = np.random.default_rng(seed)
        freqs = draw.uniform(-1, 1, 12) * rho / 2
        amps = draw.normal(size=12)
        phases = draw.uniform(0, 2 * np.pi, 12)

        def signal(t, f=freqs, a=amps, p=phases):
            return np.sum(a * np.cos(2 * np.pi * f * np.asarray(t)[..., None] + p), -1)

        query = draw.uniform(-0.5, 0.5, 50)
        taps = np.floor(query)[:, None] + np.arange(-15, 17)
        estimate = np.sum(kernel(query[:, None] - taps) * signal(taps), axis=1)
        worst = max(
            worst, np.max(np.abs(estimate - signal(query))) / np.abs(amps).sum()
        )
    return worst


def kernel_error_worst(rho, kernel=None, n_freq=61, n_offset=41):
    """Worst single-tone reconstruction error of a 32-tap kernel.

    A unit tone at frequency ``f`` cycles per sample is sampled at the
    integers and rebuilt at an offset ``q`` inside one cell. Maximized over the
    tone phase, the error is the magnitude of ``sum_k w(q - k) exp(2 pi i f k)
    - exp(2 pi i f q)``. This returns the maximum of that magnitude over every
    frequency up to ``rho / 2`` and every offset in one cell, which is the
    worst case for a signal of unit amplitude in one tone.

    Args:
        rho: Oversampling factor; Nyquist is 1.
        kernel: Function of the offset ``q - tap`` in samples; the yippy.sinc
            Kaiser-sinc weights when omitted.
        n_freq: Frequencies per sweep, from 0 to ``rho / 2``.
        n_offset: Offsets per sweep, from 0 to 1 sample.

    Returns:
        The worst absolute error for a unit-amplitude tone.
    """
    import numpy as np

    from yippy.sinc.kernel import weights

    if kernel is None:

        def kernel(t):
            return np.asarray(weights(jnp.asarray(t)))

    freqs = np.linspace(0.0, rho / 2, n_freq)
    query = np.linspace(0.0, 1.0, n_offset)
    taps = np.arange(-15, 17)
    w = kernel(query[:, None] - taps)
    tone = np.exp(2j * np.pi * freqs[:, None] * taps)
    read = tone @ w.T
    truth = np.exp(2j * np.pi * freqs[:, None] * query)
    return float(np.max(np.abs(read - truth)))
