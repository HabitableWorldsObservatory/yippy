"""Plotting stays optional and preserves caller data and axes geometry."""

import importlib
import inspect
import subprocess
import sys

import numpy as np
import pytest
from hwoutils.grids import Frame
from lod_unit import lod
from optixstuff.apertures import DEFAULT_APERTURE_RULE
from optixstuff.performance import PerformanceCurves

import yippy
from yippy import performance, viz


@pytest.fixture
def curves():
    """Two-node record derived from an aperture with constant stellar density."""
    separation = np.array([0.0, 2.0])
    throughput = np.array([0.1, 0.5])
    area = np.full(2, np.pi * DEFAULT_APERTURE_RULE.radius_lod**2)
    density = np.full(2, 1e-8)
    return PerformanceCurves(
        separation,
        Frame(500, 6, "circumscribed"),
        DEFAULT_APERTURE_RULE,
        throughput=throughput,
        throughput_variance=throughput / 2,
        core_area_lod2=area,
        core_area_variance_lod2=area / 2,
        leakage_fraction=density * area,
        leakage_fraction_variance=density * area / 2,
        leakage_density_lod2=density,
        leakage_density_variance_lod2=density,
        raw_contrast=density * area / throughput,
        iwa_lod=0.75,
        data_extent_lod=4.0,
        stellar_diameter=0.0,
    )


def test_default_curve_calls_return_data(curves, tmp_path, monkeypatch):
    """Fails if any default curve call plots, drops data or uses a missing helper.

    Basis: exact algebra and mutation. Stub the numerical producer with
    independently specified arrays and exercise default FITS persistence.
    The numerical producers have separate analytic coverage.
    """
    from yippy._yip import _YIPData

    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    monkeypatch.setitem(sys.modules, "eyepiece", None)
    from synthetic_yip import write_synthetic_yip

    package = write_synthetic_yip(tmp_path / "package")
    coro = _YIPData(package.path)
    record = coro._prepare_record()
    assert "plot" not in inspect.signature(coro._prepare_record).parameters
    assert record.rule == curves.rule
    assert list(coro._perf_dir.glob("*.fits"))
    assert not hasattr(yippy, "plot_performance_curve")
    assert not hasattr(performance, "plot_performance_curve")


def test_viz_optional_import():
    """Fails if importing base or viz eagerly loads plotting dependencies.

    Basis: mutation; blocked optional modules must allow both imports and
    produce the actionable extra hint only at the drawing boundary.
    """
    code = """
import sys
for name in ('eyepiece', 'matplotlib', 'manim', 'manim_slides'):
    sys.modules[name] = None
import yippy
import yippy.viz as viz
try:
    viz.plot_performance_curve([0, 1], [0.1, 0.5])
except ImportError as exc:
    assert "pip install 'yippy[viz]'" in str(exc)
else:
    raise AssertionError('drawing must require the optional extra')
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=5)


# tol_colors 2.2.0 builds colormaps with the pending-deprecated set_bad API.
@pytest.mark.filterwarnings(
    "ignore:The set_bad function will be deprecated.*:"
    "PendingDeprecationWarning:tol_colors"
)
def test_viz_records_arrays_and_axes(curves):
    """Fails if drawing computes science, alters data or shrinks a handed slot.

    Basis: exact array equality; 8 float64 eps for normalized axes geometry.
    Covers every record metric, the array-only transmission, and multiple
    stellar diameters. Log scales must preserve the original line values.
    """
    import eyepiece as ep
    import hwostyle
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    before = mpl.rcParams.copy()
    importlib.reload(viz)
    assert dict(mpl.rcParams) == dict(before)
    with hwostyle.light():
        fig, axes = plt.subplots(1, 2, layout="constrained")
        for metric in (
            "throughput",
            "throughput_variance",
            "core_area_lod2",
            "core_area_variance_lod2",
            "raw_contrast",
            "leakage_fraction",
            "leakage_fraction_variance",
            "leakage_density_lod2",
            "leakage_density_variance_lod2",
        ):
            result = viz.plot_performance_curve(curves, metric=metric, ax=axes[0])
            assert isinstance(result, ep.plot.PlotResult)
            assert result.ax is axes[0]
            assert set(result.artists) <= ep.plot.ARTIST_KEYS
            line = result.artists["line"]
            np.testing.assert_array_equal(line.get_xdata(), [0.0, 2.0])
            np.testing.assert_array_equal(line.get_ydata(), getattr(curves, metric))
            assert result.ax.get_ylabel()
        result = viz.plot_performance_curve(
            [0.0, 2.0], [0.0, 1.0], metric="occulter_transmission"
        )
        assert isinstance(result, ep.plot.PlotResult)
        assert result.ax.get_yscale() == "linear"
        plt.close(result.fig)
        profiles = {0.0 * lod: np.array([1e-8, 2e-8]), 0.1 * lod: [2e-8, 4e-8]}
        result = viz.plot_core_mean_intensity([0.0, 2.0], profiles, ax=axes[1])
        assert isinstance(result, ep.plot.PlotResult)
        assert set(result.artists) == {"lines"}
        assert result.ax.get_yscale() == "log"
        lines = result.artists["lines"]
        for line, expected in zip(lines, profiles.values(), strict=True):
            np.testing.assert_array_equal(line.get_ydata(), expected)
        assert lines[0].get_color() != lines[1].get_color()
        assert lines[0].get_marker() != lines[1].get_marker()
        result = viz.plot_core_mean_intensity([0, 2], {0.0: [1e-8, 2e-8]})
        plt.close(result.fig)
        fig.canvas.draw()
        np.testing.assert_allclose(
            axes[0].get_position(original=True).width,
            axes[1].get_position(original=True).width,
            rtol=8 * np.finfo(float).eps,
        )
        for args, kw in (
            ((curves,), {"metric": "invalid"}),
            ((curves, [1, 2]), {}),
            ((curves,), {"metric": "occulter_transmission"}),
            (([0, 2],), {}),
        ):
            with pytest.raises(ValueError):
                viz.plot_performance_curve(*args, **kw)
        with pytest.raises(ValueError, match="at least one"):
            viz.plot_core_mean_intensity([0, 2], {})
        plt.close(fig)
