"""A lit source cell must fit wholly inside the scene operator's input grid."""

from dataclasses import replace

import equinox as eqx
import jax
import numpy as np
from optixstuff.grids import Image, Measure

from yippy import SceneCoronagraph
from yippy.scene import AbstractSceneOperator


class _Probe(AbstractSceneOperator):
    """Accept every native cell and preserve the resampled scene in the image."""

    def render(self, flux, grid):
        """Return the flux and full native coverage."""
        return flux, np.ones(grid.shape, dtype=bool)


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


def test_render_ignores_subnormal_float32_flux_outside_grid(cached_diameter_coro):
    """Only the normal float32 sample makes the out-of-grid scene invalid."""
    coro = cached_diameter_coro
    source = coro.grid
    model = SceneCoronagraph(coro, (_Probe(),), scene_grid=source)
    ny, nx = source.shape
    larger = replace(source, shape=(ny, nx + 2), center_x=source.center_x + 1)
    tiny = np.finfo(np.float32).tiny
    subnormal = np.nextafter(tiny, np.float32(0))
    with jax.enable_x64():
        for render in (model.render_scene, eqx.filter_jit(model.render_scene)):
            for sample, valid in ((subnormal, True), (tiny, False)):
                flux = np.zeros(larger.shape, dtype=np.float32)
                flux[ny // 2, -1] = sample
                # The cutoff also applies when float64 arithmetic preserves
                # the subnormal float32 input rather than flushing it to zero.
                for data in (flux, flux.astype(np.float64)):
                    scene = Image(data, larger, (), Measure.integral(), "flux")
                    assert bool(render(scene, None, grid=source).valid) is valid
