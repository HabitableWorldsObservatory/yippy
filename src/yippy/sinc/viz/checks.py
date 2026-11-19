"""Logarithmic check rails with explicit exact-zero evidence."""

import numpy as np

from yippy.sinc.viz import _require
from yippy.sinc.viz._common import single_ax


def plot_check_residuals(
    receipt=None,
    *,
    independent=None,
    mirror=None,
    axis=None,
    target=1e-5,
    mirror_tolerance=1e-6,
    ax=None,
):
    """Compare independent, mirror, and axis evidence on three horizontal rails.

    Args:
        receipt: Optional Receipt supplying local-peak and axis evidence.
        independent: Array overriding receipt.empirical_local_peak.
        mirror: Array overriding receipt.mirror_local_peak.
        axis: Array overriding receipt.axis_relative and its exact-pass flags.
        target: Positive threshold on the independent rail.
        mirror_tolerance: Positive threshold on the mirror and axis rails.
        ax: Single axes; None creates a figure.

    Returns:
        PlotResult with ``scatter`` holding nonzero points then exact-zero
        markers for each rail (six collections total), ``lines`` holding
        three rail baselines then three threshold lines, and ``text`` holding
        the threshold labels. Zero points and exact axis passes use left-edge
        triangles. Rail labels count exact zeros and skipped NaNs. An exact
        axis pass counts as zero even when its relative residual is undefined.

    Raises:
        ValueError: Invalid thresholds, infinite residuals, inconsistent axis
            evidence lengths, or an axes array instead of one axes.

    Example::

        from yippy.sinc.viz import plot_check_residuals
        result = plot_check_residuals(independent=[0, 1e-6, 2e-5])
    """
    values = [
        np.asarray(
            explicit if explicit is not None else getattr(receipt, field, ()),
            dtype=float,
        ).ravel()
        for explicit, field in (
            (independent, "empirical_local_peak"),
            (mirror, "mirror_local_peak"),
            (axis, "axis_relative"),
        )
    ]
    exact = np.asarray(
        getattr(receipt, "axis_exact", ()) if axis is None else (), dtype=bool
    ).ravel()
    if exact.size and exact.size != values[2].size:
        raise ValueError("axis_exact and axis_relative must have matching lengths")
    if exact.size:
        values[2] = np.where(exact, 0.0, values[2])
    target, mirror_tolerance = float(target), float(mirror_tolerance)
    if not all(np.isfinite(t) and t > 0 for t in (target, mirror_tolerance)):
        raise ValueError("target and mirror_tolerance must be finite and positive")
    if any(np.any(np.isinf(v)) for v in values):
        raise ValueError("residuals must be finite or NaN")
    values = [np.abs(v) for v in values]
    positive = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in values])
    minimum = min(
        target, mirror_tolerance, float(positive.min()) if positive.size else target
    )
    floor = max(minimum * 0.1, np.nextafter(0.0, 1.0))
    upper = max(
        target, mirror_tolerance, float(positive.max()) if positive.size else target
    )
    ep = _require.eyepiece()
    ax = single_ax(ax)
    ax.set_xscale("log")
    ax.set_xlim(floor, upper * 10)
    ax.set_ylim(2.5, -0.8)
    points, lines, texts, labels = [], [], [], []
    names = ("independent checks", "mirror checks", "axis invariance")
    for row, (name, data) in enumerate(zip(names, values, strict=True)):
        nonzero = data[np.isfinite(data) & (data > 0)]
        zeros = int(np.count_nonzero(data == 0))
        skipped = int(np.count_nonzero(np.isnan(data)))
        suffix = []
        if zeros:
            suffix.append(f"{zeros} exactly 0")
        if skipped:
            suffix.append(f"{skipped} NaN skipped")
        if data.size == 0:
            suffix.append("none")
        labels.append(name + (f" ({', '.join(suffix)})" if suffix else ""))
        lines.append(
            ax.plot(
                [floor, upper * 10], [row, row], color=ep.style.neutral(0.2), zorder=0
            )[0]
        )
        points.append(
            ax.scatter(
                nonzero,
                np.full(nonzero.size, row),
                s=25,
                color=ep.style.role("data"),
                zorder=3,
            )
        )
        points.append(
            ax.scatter(
                np.full(zeros, floor),
                np.full(zeros, row),
                marker=">",
                s=35,
                color=ep.style.role("data"),
                clip_on=False,
                zorder=3,
            )
        )
    for row, value in enumerate((target, mirror_tolerance, mirror_tolerance)):
        lines.append(
            ax.plot(
                [value, value],
                [row - 0.3, row + 0.3],
                color=ep.style.neutral(0.45),
                linestyle="--",
            )[0]
        )
        label = "target" if row == 0 else "tolerance"
        texts.append(
            ax.text(
                value,
                row - 0.32,
                f"{label} " + f"{value:.0e}".replace("e-0", "e-"),
                ha="center",
                va="bottom",
                color=ep.style.neutral(0.45),
                bbox=ep.style.backing(),
            )
        )
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("max |error| / local peak")
    return ep.plot.PlotResult(ax, {"scatter": points, "lines": lines, "text": texts})
