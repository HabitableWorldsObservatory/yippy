"""Prefiltered reads against direct propagation of a vortex coronagraph."""

import jax
import jax.numpy as jnp
import pytest
from optixstuff.grids import Frame, PixelGrid

import yippy.sinc as sc

PUPIL_SAMPLES, PIXEL_LOD, DETECTOR = 128, 0.125, 48
SPACING, NODES = 0.3, 33


@pytest.mark.slow(reason="about a thousand coronagraph propagations, 33 s")
def test_reads_match_direct_vortex_propagation():
    """Fails if prefiltered reads of a real coronagraph leave their design scale.

    A charge-6 vortex with a 90 percent Lyot stop is propagated on a 33 x 33
    lattice of source offsets at 0.3 lambda/D, off axis so the table is not
    mirror-symmetric, and again at eight off-lattice offsets. The entrance
    aperture has unit diameter, so the intensity is band-limited to one cycle
    per lambda/D and the sampling ratio is 0.6. Measured worst errors relative
    to each image's own peak are 1.8e-5 at 7 taps and 1.6e-6 at 8 taps; the
    limits leave a factor of two to three. Slow tier: about a thousand
    propagations.
    """
    # Imported here so that collecting the quick tier does not need physicaloptix.
    pytest.importorskip("physicaloptix")
    from physicaloptix import (
        Field,
        Fraunhofer,
        MultiScaleVortex,
        OpticalPath,
        PlaneKind,
        Stage,
    )
    from physicaloptix.elements import SampledOptic

    pupil = PixelGrid.pupil(PUPIL_SAMPLES)
    focal = PixelGrid.focal(DETECTOR, PIXEL_LOD, Frame(500.0, 6.0, "circumscribed"))
    u, v = pupil.coordinates()
    aperture = (jnp.hypot(u, v) <= 0.5).astype(jnp.float64)
    vortex = MultiScaleVortex.build(
        charge=6, npup=PUPIL_SAMPLES, q=64, scaling_factor=4, window_size=16
    )
    lyot = SampledOptic(
        transmission=(jnp.hypot(u, v) <= 0.45).astype(jnp.float64),
        grid=pupil,
        plane=PlaneKind.PUPIL,
    )
    path = OpticalPath(
        stages=(
            Stage("vortex", vortex),
            Stage("lyot", lyot),
            Stage("science", Fraunhofer(grid_in=pupil, grid_out=focal)),
        )
    )

    def image(source):
        tilt = jnp.exp(2j * jnp.pi * (source[0] * u + source[1] * v))
        field = Field(data=aperture * tilt, grid=pupil, plane=PlaneKind.PUPIL)
        amplitude = path.propagate(field)[0].data
        return amplitude.real**2 + amplitude.imag**2

    propagate = jax.jit(lambda sources: jax.lax.map(image, sources, batch_size=16))
    center = jnp.array([3.0, 1.5])
    iy, ix = jnp.indices((NODES, NODES), dtype=jnp.int32)
    indices = jnp.stack((ix.ravel(), iy.ravel()), axis=1)
    origin = center - SPACING * (NODES - 1) / 2
    offsets = origin + indices * SPACING
    samples = propagate(offsets)
    unit = float(jnp.max(samples))
    shape = (DETECTOR, DETECTOR)
    products = sc.Products(
        PixelGrid(
            shape,
            (jnp.full(2, (DETECTOR - 1) / 2))[0],
            (jnp.full(2, (DETECTOR - 1) / 2))[1],
            jnp.eye(2) * PIXEL_LOD,
            "lod",
            Frame(500.0, 6.0, "circumscribed"),
        ),
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
        "none",
        jnp.full(2, False),
        "declaration",
        "argument",
        jnp.array(1, dtype=jnp.int32),
        "checks",
        jnp.array([False]),
    )
    domain = sc.Domain(
        "box", "closed", jnp.stack([center - 1.5, center + 1.5]), jnp.zeros(2)
    )
    memory = sc.MemoryBudget(10**9, 10**8, 10**9, 4 * 10**9)
    queries = center + jax.random.uniform(
        jax.random.key(0), (8, 2), minval=-1.4, maxval=1.4
    )
    direct = propagate(queries) / unit
    peak = jnp.max(direct, axis=(1, 2))
    worst = {}
    for taps in (7, 8):
        profile = sc.PrefilteredKernelProfile(taps=taps)
        grid = sc.admit_grid(
            samples[None] / unit,
            offsets,
            declared_lattice=sc.source_grid(
                origin,
                jnp.full(2, SPACING),
                (NODES, NODES),
                Frame(500.0, 6.0, "circumscribed"),
            ),
            indices_xy=indices,
            products=products,
            reflection=reflection,
            domain=domain,
            profile=profile,
            memory=memory,
        ).require()
        reader = sc.PrefilteredReader(
            grid, domain, profile, sc.ReadPolicy(), sc.Tiling(query_chunk=4), memory
        )
        reads = sc.require_read(
            sc.read_batch(reader, queries, jnp.zeros(8, dtype=jnp.int32))
        )
        error = jnp.max(jnp.abs(reads.intensity - direct), axis=(1, 2))
        worst[taps] = float(jnp.max(error / peak))
    assert worst[7] <= 4e-5
    assert worst[8] <= 5e-6
    assert worst[8] <= worst[7] / 3
