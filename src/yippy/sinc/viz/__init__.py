"""Stateless views of yippy.sinc types and arrays, built on eyepiece.

Install ``yippy[viz]`` to use the lazily exported plotting functions.
Importing this package does not load eyepiece or activate a style. Each
function accepts one ``ax`` (never an axes array) and returns a PlotResult.
"""

_LAZY = {
    "plot_kernel": "kernel",
    "plot_stencil": "kernel",
    "plot_stencil_2d": "stencil2d",
    "plot_spacing_rule": "rule",
    "plot_coverage": "coverage",
    "plot_read": "images",
    "plot_residual": "images",
    "plot_check_residuals": "checks",
    "plot_band_spectrum": "spectra",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        from yippy.sinc.viz import _require

        _require.eyepiece()
        module = importlib.import_module(f"yippy.sinc.viz.{_LAZY[name]}")
        return getattr(module, name)
    raise AttributeError(f"module 'yippy.sinc.viz' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals()) | set(__all__))
