"""Builders for sinc tests: synthetic deliveries, readers and band-limited truths."""

import jax
import jax.numpy as jnp
from optixstuff.grids import Frame, PixelGrid

import yippy.sinc as sc
from yippy.sinc.geometry import origin_xy, spacing_xy


def analytic(source, asymmetric=False, shape=(8, 6)):
    rows, columns = jnp.indices(shape, dtype=jnp.float64)
    xpix = (columns - (shape[1] - 1) / 2) * 0.25
    ypix = (rows - (shape[0] - 1) / 2) * 0.25
    x = jnp.cos(2 * jnp.pi * 0.16 * (source[0] - xpix))
    y = jnp.cos(2 * jnp.pi * 0.12 * (source[1] - ypix))
    image = 2 + 0.4 * x + 0.3 * y + 0.1 * x * y
    if asymmetric:
        image = image + 0.3 * jnp.sin(2 * jnp.pi * 0.1 * source[0]) + 0.04 * xpix
    return image


def delivery(
    quadrant=False,
    asymmetric=False,
    half=20,
    shape=(8, 6),
    spacing=0.25,
    truth=None,
    limit=1.0,
):
    """Build admission inputs; ``truth`` maps a source [2] to an image of ``shape``."""
    if truth is None:

        def truth(q):
            return analytic(q, asymmetric, shape)

    # ``half`` and ``spacing`` are scalars or (x, y) pairs.
    half = jnp.broadcast_to(jnp.asarray(half), (2,))
    spacing = jnp.broadcast_to(jnp.asarray(spacing, dtype=jnp.float64), (2,))
    count = tuple(int(n) + 1 if quadrant else 2 * int(n) + 1 for n in half)
    origin = jnp.zeros(2) if quadrant else -half * spacing
    lattice = sc.source_grid(
        origin, spacing, (count[1], count[0]), Frame(500.0, 6.0, "circumscribed")
    )
    iy, ix = jnp.indices(lattice.shape, dtype=jnp.int32)
    indices = jnp.stack((ix.ravel(), iy.ravel()), axis=1)
    offsets = origin_xy(lattice) + indices * spacing_xy(lattice)
    images = jax.vmap(truth)(offsets)[None]
    pixels = PixelGrid(
        shape,
        (jnp.array([(shape[1] - 1) / 2, (shape[0] - 1) / 2]))[0],
        (jnp.array([(shape[1] - 1) / 2, (shape[0] - 1) / 2]))[1],
        jnp.eye(2) * 0.25,
        "lod",
        Frame(500.0, 6.0, "circumscribed"),
    )
    products = sc.Products(
        pixels,
        sc.Normalization(
            "incident_photon", jnp.array([1.0]), jnp.array([jnp.nan]), (), "units"
        ),
        "monochromatic",
        jnp.array([500.0]),
        jnp.array([[500.0, 500.0]]),
        jnp.array([500.0]),
        jnp.array(jnp.nan),
        jnp.ones((1, 2)),
        jnp.ones(1, dtype=jnp.int32),
        ("support",),
        sc.Premises(
            jnp.ones(6, dtype=bool), jnp.ones(6, dtype=jnp.int32), ("truth",) * 6
        ),
        "optics",
        "mono",
        "synthetic",
    )
    reflection = sc.Reflection(
        "two_axis_pixel_flip_v1" if quadrant else "none",
        jnp.full(2, quadrant),
        "declaration",
        "argument",
        jnp.array(1, dtype=jnp.int32),
        "checks",
        jnp.array([False]),
    )
    domain = sc.Domain(
        "box", "closed", jnp.array([[-limit, -limit], [limit, limit]]), jnp.zeros(2)
    )
    mirrors = None
    if quadrant:
        signed = jnp.array([[-1, 1], [1, -1], [-1, -1]], dtype=jnp.int32)
        positions = signed * spacing
        mirrors = sc.MirrorCheckSet(
            positions,
            signed,
            jax.vmap(truth)(positions),
            jnp.zeros(3, dtype=jnp.int32),
            "independent",
        )
    return dict(
        intensity_nodes=images,
        offsets_xy_lod=offsets,
        declared_lattice=lattice,
        indices_xy=indices,
        products=products,
        reflection=reflection,
        domain=domain,
        memory=sc.MemoryBudget(100_000_000, 10_000_000, 10_000_000, 400_000_000),
        mirror_checks=mirrors,
    )


def tone_truth(shape=(6, 5), symmetric=True, terms=6, seed=0):
    """Return a band-limited truth q -> image and the l1 norm of its spectrum.

    Every pixel is a constant plus products of cosines in the two source
    coordinates with frequencies up to exactly 1 cycle per lambda/D, the band
    edge that ``delivery`` declares (unit support extent at the reference
    wavelength). The symmetric form satisfies the two-axis detector-flip
    symmetry exactly; the other adds phases that break it.
    """
    keys = jax.random.split(jax.random.key(seed), 4)
    nu = jax.random.uniform(keys[0], (terms, 2), minval=0.2, maxval=1.0)
    nu = nu.at[0].set(1.0).at[1, 0].set(1.0).at[2, 1].set(1.0)
    amplitude = jax.random.uniform(keys[1], (terms,), minval=0.2, maxval=1.0)
    phase = jax.random.uniform(keys[2], (terms, 2), maxval=2 * jnp.pi)
    phase = phase * (0.0 if symmetric else 1.0)
    rows, columns = jnp.indices(shape, dtype=jnp.float64)
    xpix = (columns - (shape[1] - 1) / 2) * 0.37
    ypix = (rows - (shape[0] - 1) / 2) * 0.41
    floor = float(jnp.sum(amplitude)) + 0.5

    def truth(q):
        x = jnp.cos(
            2 * jnp.pi * nu[:, 0, None, None] * (q[0] - xpix) + phase[:, 0, None, None]
        )
        y = jnp.cos(
            2 * jnp.pi * nu[:, 1, None, None] * (q[1] - ypix) + phase[:, 1, None, None]
        )
        return floor + jnp.sum(amplitude[:, None, None] * x * y, axis=0)

    return truth, floor + float(jnp.sum(amplitude))


def tone_delivery(
    quadrant=False, symmetric=True, half=(22, 20), shape=(6, 5), spacing=(0.3, 0.28)
):
    """A band-edge delivery with unequal axes and a prefilter-sized budget.

    Sampling ratios are 0.6 (x) and 0.56 (y); the table and the detector are
    not square, so an axis or flip mix-up cannot cancel.
    """
    truth, norm = tone_truth(shape, symmetric)
    data = delivery(
        quadrant, half=half, shape=shape, spacing=spacing, truth=truth, limit=2.0
    )
    data["memory"] = sc.MemoryBudget(100_000_000, 10_000_000, 60_000_000, 400_000_000)
    return data, truth, norm


def make_prefiltered(data, taps=7, tiling=None):
    profile = sc.PrefilteredKernelProfile(taps=taps)
    admission = sc.admit_grid(**data, profile=profile)
    reader = sc.PrefilteredReader(
        admission.require(),
        data["domain"],
        profile,
        sc.ReadPolicy(),
        sc.Tiling(query_chunk=2) if tiling is None else tiling,
        data["memory"],
    )
    return reader, admission.receipt


def make_reader(data):
    admission = sc.admit_grid(**data)
    grid = admission.require()
    return sc.Interpolator(
        grid,
        data["domain"],
        sc.KernelProfile(),
        sc.ReadPolicy(),
        sc.Tiling(pixel_tile=13, query_chunk=2),
        data["memory"],
    )
