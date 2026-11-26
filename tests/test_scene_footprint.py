"""A lit source cell must fit wholly inside the scene operator's input grid."""

from dataclasses import replace
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from optixstuff.grids import Image, Measure, PixelGrid

from yippy import SceneCoronagraph
from yippy.scene import AbstractSceneOperator, _DenseSceneOperator


class _Probe(AbstractSceneOperator):
    """Accept every native cell and preserve the resampled scene in the image."""

    def render(self, flux, grid):
        """Return the flux and full native coverage."""
        return flux, np.ones(grid.shape, dtype=bool)


def test_covered_scene_tangent_at_zero_matches_unit_response(cached_diameter_coro):
    """A linear dense operator has its unit response as the JVP from zero.

    Exact algebra: unit PSFs sum unit cell fluxes, with exact overlap for
    the aligned two-by-two refinement. Both paths then sample the same image.
    """
    coro = cached_diameter_coro
    source = replace(coro.grid, shape=(2, 2), center_x=0.5, center_y=0.5)
    operator = _DenseSceneOperator(
        jnp.ones(source.shape + coro.grid.shape), jnp.ones(source.shape, dtype=bool)
    )
    model = SceneCoronagraph(coro, (operator,), scene_grid=source)
    fine = PixelGrid.regular((4, 4), source.step / 2, source.unit, frame=source.frame)

    def render(flux, grid):
        scene = Image(flux, grid, (), Measure.integral(), "flux")
        return model.render_scene(scene, None, grid=coro.grid).image.data

    for grid in (source, fine):
        unit = jnp.ones(grid.shape)
        for run in (render, eqx.filter_jit(render)):
            expected = run(unit, grid)
            assert jnp.any(expected > 0)
            _, tangent = jax.jvp(
                partial(run, grid=grid), (jnp.zeros_like(unit),), (unit,)
            )
            np.testing.assert_array_equal(tangent, expected)


def test_render_checks_lit_input_cell_corners(cached_diameter_coro):
    """Cell boundaries determine support, including for a grid with dark padding."""
    coro = cached_diameter_coro
    source = coro.grid
    model = SceneCoronagraph(coro, (_Probe(),), scene_grid=source)
    ny, nx = source.shape
    # At column nx the center is 1/4 pitch inside, but its right edge is outside.
    larger = replace(source, shape=(ny, nx + 2), center_x=source.center_x + 0.75)
    render = eqx.filter_jit(model.render_scene)
    for column, valid in ((nx - 2, True), (nx, False), (nx + 1, False)):
        flux = np.zeros(larger.shape)
        flux[ny // 2, column] = 1
        scene = Image(flux, larger, (), Measure.integral(), "flux")
        result = render(scene, None, grid=source)
        assert bool(result.valid) is valid, column
        if column == nx + 1:
            np.testing.assert_array_equal(result.image.data, 0)
        if valid:
            assert np.asarray(result.image.data).sum() > 0


def test_render_ignores_subnormal_float32_flux_outside_grid(
    cached_diameter_coro, precision
):
    """Only the normal float32 sample makes the out-of-grid scene invalid."""
    coro = cached_diameter_coro
    source = coro.grid
    model = SceneCoronagraph(coro, (_Probe(),), scene_grid=source)
    ny, nx = source.shape
    larger = replace(source, shape=(ny, nx + 2), center_x=source.center_x + 1)
    tiny = np.finfo(np.float32).tiny
    subnormal = np.nextafter(tiny, np.float32(0))
    for render in (model.render_scene, eqx.filter_jit(model.render_scene)):
        for sample, valid in ((subnormal, True), (tiny, False)):
            flux = np.zeros(larger.shape, dtype=np.float32)
            flux[ny // 2, -1] = sample
            # The cutoff also applies when float64 arithmetic preserves
            # the subnormal float32 input rather than flushing it to zero.
            for dtype in {np.dtype("float32"), precision}:
                scene = Image(
                    flux.astype(dtype), larger, (), Measure.integral(), "flux"
                )
                assert bool(render(scene, None, grid=source).valid) is valid


class _UncoveredProbe(AbstractSceneOperator):
    """Expose whether dark values reach an uncovered native cell."""

    def render(self, flux, grid):
        """Pass flux through and declare every cell uncovered."""
        return flux, np.zeros(grid.shape, dtype=bool)


def test_render_applies_the_same_dark_rule_to_uncovered_native_cells(
    cached_diameter_coro, precision
):
    """Float32 subnormal flux is removed before coverage and rendering."""
    coro = cached_diameter_coro
    source = coro.grid
    model = SceneCoronagraph(coro, (_UncoveredProbe(),), scene_grid=source)
    render = eqx.filter_jit(model.render_scene)
    tiny = np.finfo(np.float32).tiny
    subnormal = np.nextafter(tiny, np.float32(0))
    # A binary64 value just below the cutoff is dark, but a binary32 scene
    # stores it as the cutoff itself, which is lit.
    just_below = np.nextafter(float(tiny), 0.0)
    lit_below = np.dtype(precision) == np.float32
    for sample, valid in (
        (subnormal, True),
        (just_below, not lit_below),
        (tiny, False),
    ):
        flux = np.zeros(source.shape, dtype=precision)
        flux[source.shape[0] // 2, source.shape[1] // 2] = sample
        result = render(
            Image(flux, source, (), Measure.integral(), "flux"), None, grid=source
        )
        assert bool(result.valid) is valid
        if valid:
            np.testing.assert_array_equal(result.image.data, 0)


def test_render_removes_dark_cells_before_coarsening(cached_diameter_coro, precision):
    """Four dark fine cells stay dark even when added into one native cell."""
    coro = cached_diameter_coro
    source = coro.grid
    model = SceneCoronagraph(coro, (_UncoveredProbe(),), scene_grid=source)
    fine = PixelGrid.regular(
        tuple(2 * n for n in source.shape),
        source.step / 2,
        source.unit,
        frame=source.frame,
        center=(2 * source.center_x + 0.5, 2 * source.center_y + 0.5),
    )
    flux = np.zeros(fine.shape, dtype=precision)
    flux[2:4, 2:4] = np.nextafter(np.finfo(np.float32).tiny, np.float32(0))
    scene = Image(flux, fine, (), Measure.integral(), "flux")
    result = eqx.filter_jit(model.render_scene)(scene, None, grid=source)
    assert result.valid
    np.testing.assert_array_equal(result.image.data, 0)
