"""Notebook frame measurements must match the public performance curves."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lod_unit import lod
from test_band_limited_apertures import image_coro

from yippy import HalfMaximum
from yippy import performance as perf


@pytest.mark.parametrize(
    "prefix,cell_index,frames,metric",
    [
        ("01", 6, "frames", "throughput"),
        ("01", 13, "trunc_frames", "throughput"),
        ("02", 7, "contrast_frames", "contrast"),
        ("03", 8, "area_frames", "area"),
    ],
)
def test_notebook_measurements_match_public_curves(prefix, cell_index, frames, metric):
    """Run the actual frame-building code on a faint core plus a bright lobe."""
    path = next(Path("docs/examples").glob(prefix + "_*.ipynb"))
    source = "".join(json.loads(path.read_text())["cells"][cell_index]["source"])
    source = source.split("# Compute global colorscale")[0].split("# 3-row")[0]
    y, x = np.indices((32, 32))
    psf = 0.001 * np.exp(-((y - 16) ** 2 + (x - 16) ** 2) / 4)
    psf += np.exp(-((y - 16) ** 2 + (x - 4) ** 2) / 4)
    coro = image_coro(psf, 0.25, star=np.ones_like(psf))
    namespace = {name: getattr(perf, name) for name in dir(perf)}
    namespace.update(np=np, jnp=jnp, jax=jax, coro=coro, lod=lod)
    namespace["positions"] = list(perf._iter_xaxis_positions(coro))
    with jax.enable_x64():
        exec(compile(source, str(path), "exec"), namespace)
    frame = namespace[frames][0]
    if frames == "frames":
        expected = perf.compute_throughput_curve(coro)[1][0]
    elif metric == "contrast":
        expected = perf.compute_raw_contrast_curve(coro)[1][0]
    else:
        _, flux, area = perf.compute_truncation_curves(coro, namespace["aperture_rule"])
        expected = area[0] if metric == "area" else flux[0]
    assert float(frame[metric]) == pytest.approx(expected, rel=1e-6)
    if "mask" in frame:
        weights = frame["mask"]
        assert np.any((weights > 0) & (weights < 1))


def test_comparison_notebook_uses_nominal_position():
    """The comparison and optimization sweeps use the same core as curves."""
    path = next(Path("docs/examples").glob("05_*.ipynb"))
    source = "".join(json.loads(path.read_text())["cells"][6]["source"])
    source = source.split("positions = ")[0]
    y, x = np.indices((32, 32))
    psf = 0.001 * np.exp(-((y - 16) ** 2 + (x - 16) ** 2) / 4)
    psf += np.exp(-((y - 16) ** 2 + (x - 4) ** 2) / 4)
    coro = image_coro(psf, 0.25)
    namespace = {name: getattr(perf, name) for name in dir(perf)}
    namespace.update(np=np, jnp=jnp, jax=jax, coro=coro)
    exec(compile(source, str(path), "exec"), namespace)
    pos = next(perf._iter_xaxis_positions(coro))
    flux, area = namespace["trunc_core"](pos, 0.3)
    _, expected_flux, expected_area = perf.compute_truncation_curves(
        coro, HalfMaximum(0.3)
    )
    np.testing.assert_allclose(
        [flux, area], [expected_flux[0], expected_area[0]], rtol=1e-6
    )
