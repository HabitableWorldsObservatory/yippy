"""The 2D blend read follows a radial throughput ramp between rings of nodes.

The shipped two-dimensional deliveries tabulate nodes on rings of constant
radius. A source between two rings must read a throughput between the two ring
values, rising with radius, rather than the value of whichever ring sits
closest in an inverse-distance sense.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yippy import open_yip

FRACTIONS = np.array([0.25, 0.5, 0.75])
RING_LO, RING_HI = 1.6, 2.9
# The spike measured the radial-linear miss at 2.6 percent on this ramp at the
# quarter point, against 11.6 percent for the previous four-nearest read.
RAMP_TOL = 0.03


@pytest.mark.parametrize("azimuth_deg", [0.0, 15.0, 45.0])
def test_a_source_between_rings_reads_the_analytic_throughput_ramp(
    azimuth_deg, synthetic_rings
):
    """A source between rings reads T(r) = r^2 / (r^2 + a^2) and rises with r.

    The package has rings at 1.0, 1.6 and 2.9 lambda/D with four azimuths and
    Gaussian cores, so the stored throughput is the analytic ramp and the core
    shape is shift invariant: the summed read of a source is its throughput
    times the same constant as a stored node. Sources at a quarter, a half and
    three quarters of the gap between the 1.6 and 2.9 rings, on and between
    azimuth nodes, must read within 3 percent of the analytic value (basis:
    spike, radial-linear miss 2.6 percent) and must increase strictly with
    radius. The previous four-nearest inverse-distance read took its nodes from
    the 1.0 ring at the quarter point and missed by 11.6 percent; a 1.05
    weight mutant also fails the tolerance.

    Fails if the read saturates at a nearer ring (a stair) or leaves the ramp.
    """
    backend = open_yip(synthetic_rings.path).prepare()._offaxis_backend
    radius = RING_LO + FRACTIONS * (RING_HI - RING_LO)
    phi = np.radians(azimuth_deg)
    with jax.enable_x64():
        sums = np.asarray(
            backend.create_psfs(
                jnp.asarray(radius * np.cos(phi)), jnp.asarray(radius * np.sin(phi))
            ).data.sum(axis=(-1, -2))
        )
        node = float(
            backend.create_psfs(jnp.asarray([RING_LO]), jnp.asarray([0.0]))
            .data.sum(axis=(-1, -2))
            .item()
        )
    expected = synthetic_rings.occulter(radius) / synthetic_rings.occulter(RING_LO)
    np.testing.assert_allclose(sums / node, expected, rtol=RAMP_TOL, atol=0.0)
    assert np.all(np.diff(sums) > 0)
