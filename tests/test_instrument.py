"""YIP table models compose with the shared optical hardware path."""

import equinox as eqx
import jax.numpy as jnp
import optixstuff as ox
import pytest


@pytest.fixture(scope="session")
def instrument(cached_diameter_coro):
    """Selected instrument with pupil, coronagraph, detector, and a filter."""
    # The pupil takes the coronagraph's diameter: its tables fix lambda/D.
    pupil = ox.SimplePupil(
        circumscribed_diameter_m=cached_diameter_coro.grid.frame.diameter_m,
        collecting_area_m2=3.141592653589793 * 9 * (1 - 0.14**2),
    )
    detector = ox.Detector(
        pixel_scale_arcsec=0.010,
        shape=(100, 100),
        qe=0.9,
        dark_current_e_per_pixel_s=3e-5,
        read_noise_e_rms=0.0,
        clock_induced_charge_e_per_pixel_frame=1.3e-3,
        read_time_s=1000.0,
        survival_fraction=1.0,
    )
    optics_filter = ox.ConstantThroughput(throughput=0.5, name="optics")
    observatory = ox.Observatory(
        ox.Telescope(pupil),
        (
            ox.Instrument(
                "imager",
                coronagraph=cached_diameter_coro,
                attenuating_elements=(optics_filter,),
                detector=detector,
            ),
        ),
    )
    return observatory.select("imager")


# ---------------------------------------------------------------------------
# Coronagraph tests
# ---------------------------------------------------------------------------


class TestCoronagraph:
    """Verify sampled Coronagraph implements the abstract table contract."""

    def test_is_abstract_coronagraph(self, cached_diameter_coro):
        """The loaded table satisfies the shared coronagraph contract."""
        assert isinstance(cached_diameter_coro, ox.AbstractCoronagraph)

    def test_is_equinox_module(self, cached_diameter_coro):
        """The loaded table can participate in Equinox transformations."""
        assert isinstance(cached_diameter_coro, eqx.Module)

    def test_pixel_scale_positive(self, cached_diameter_coro):
        """Native angular sampling preserves its positive FITS pitch."""
        assert cached_diameter_coro.grid.step > 0

    def test_iwa_owa_ordering(self, cached_diameter_coro):
        """The measured inner working angle lies inside the sampled extent."""
        assert (
            cached_diameter_coro.curves().iwa_lod
            < cached_diameter_coro.curves().data_extent_lod
        )

    def test_throughput_in_range(self, cached_diameter_coro):
        """The synthetic aperture passes a physical flux fraction."""
        t = cached_diameter_coro.curves().evaluate("throughput", 5.0)
        assert 0.0 <= float(t) <= 1.0

    def test_core_area_positive(self, cached_diameter_coro):
        """The source aperture retains positive angular area."""
        a = cached_diameter_coro.curves().evaluate("core_area_lod2", 5.0)
        assert float(a) > 0

    def test_leakage_density_lod2_nonnegative(self, cached_diameter_coro):
        """The stellar density remains nonnegative."""
        cmi = cached_diameter_coro.curves().evaluate("leakage_density_lod2", 5.0, 0.0)
        assert float(cmi) >= 0

    def test_uniform_sky_response_in_range(self, cached_diameter_coro):
        """The native sky map retains dimensionless transmission."""
        st = cached_diameter_coro.uniform_sky_response(5.0)
        assert 0.0 <= float(st) <= 1.0


# ---------------------------------------------------------------------------
# Instrument integration tests
# ---------------------------------------------------------------------------


class TestInstrumentIntegration:
    """Verify a table-backed Instrument computes system throughput."""

    def test_system_throughput_value(self, instrument):
        """Exact algebra: the path retains the explicit 50 percent filter."""
        t = instrument.system_throughput(500.0)
        assert t == pytest.approx(0.5)

    def test_pupil_area(self, instrument):
        """Floating-point: the explicit collecting area equals the annular area."""
        expected = jnp.pi * 3.0**2 * (1.0 - 0.14**2)
        assert instrument.pupil.collecting_area_m2 == pytest.approx(
            float(expected), rel=1e-6
        )
