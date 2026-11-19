"""Hann-windowed spectra on the spatial frequency coordinate."""

import numpy as np
from matplotlib.cbook import normalize_kwargs
from matplotlib.lines import Line2D

from yippy.sinc.viz import _require
from yippy.sinc.viz._common import single_ax


def plot_band_spectrum(trace, step, *, bandwidth, ax=None, line_kw=None):
    """Draw a normalized Hann-windowed amplitude spectrum and the pupil band.

    Args:
        trace: Finite one-dimensional uniformly sampled values, at least three.
        step: Positive sample spacing in lambda/D.
        bandwidth: Highest frequency of the band, in cycles per lambda/D;
            the shaded band is ``|f| <= bandwidth``.
        ax: Single axes; None creates a figure. Updates keep the initial limits.
        line_kw: Overrides for the spectrum Line2D.

    Returns:
        PlotResult with ``line``, ``fill`` for the band, ``text`` for its
        label, and ``update(trace)`` replacing the existing line's data.
        Amplitudes are normalized to their maximum and floored at 1e-12;
        an all-zero trace is drawn entirely at the floor.

    Raises:
        ValueError: Invalid trace, spacing, bandwidth, or axes shape.

    Example::

        import numpy as np
        from yippy.sinc.viz import plot_band_spectrum
        result = plot_band_spectrum(np.cos(np.arange(128)), 0.25, bandwidth=1)
        result.update(np.sin(np.arange(128)))
    """
    trace = np.asarray(trace)
    step, bandwidth = float(step), float(bandwidth)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    if not np.isfinite(bandwidth) or bandwidth < 0:
        raise ValueError("bandwidth must be finite and nonnegative")

    def spectrum(values):
        values = np.asarray(values)
        if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
            raise ValueError(
                "trace must be finite, one-dimensional, with at least 3 samples"
            )
        amplitude = np.abs(
            np.fft.fftshift(np.fft.fft(values * np.hanning(values.size)))
        )
        maximum = float(amplitude.max())
        normalized = amplitude / maximum if maximum > 0 else amplitude
        return np.fft.fftshift(np.fft.fftfreq(values.size, step)), np.maximum(
            normalized, 1e-12
        )

    frequency, amplitude = spectrum(trace)
    ep = _require.eyepiece()
    ax = single_ax(ax)
    line = ax.plot(
        frequency,
        amplitude,
        **{"color": ep.style.role("data"), **normalize_kwargs(line_kw, Line2D)},
    )[0]
    band = ax.axvspan(-bandwidth, bandwidth, color=ep.style.neutral(0.2), zorder=0)
    text = ax.text(
        0,
        0.02,
        "pupil band",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        color=ep.style.neutral(0.45),
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1.2)
    ax.set_xlabel(r"frequency (cycles per $\lambda/D$)")
    ax.set_ylabel("normalized amplitude")

    def update(trace):
        frequency, amplitude = spectrum(trace)
        line.set_data(frequency, amplitude)

    return ep.plot.PlotResult(ax, {"line": line, "fill": band, "text": text}, update)
