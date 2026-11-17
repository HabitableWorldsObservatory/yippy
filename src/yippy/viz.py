"""Stateless performance plots; install ``yippy[viz]`` to draw them.

Style activation and figure saving belong to the caller. Importing this module
does not load eyepiece or Matplotlib or change the active style.
"""

from optixstuff.performance import PerformanceCurves

__all__ = ["plot_core_mean_intensity", "plot_performance_curve"]

_METRICS = {
    "throughput": ("Throughput", False),
    "core_area_lod2": (r"Core area [$(\lambda/D)^2$]", False),
    "leakage_fraction": ("Stellar leakage fraction", True),
    "leakage_density_lod2": (r"Stellar leakage density [$(\lambda/D)^{-2}$]", True),
    "raw_contrast": ("Raw contrast", True),
    "occulter_transmission": ("Occulter transmission", False),
}


def _eyepiece():
    """Load the optional plotting dependency at the drawing boundary."""
    try:
        import eyepiece as ep
    except ImportError:
        raise ImportError(
            "yippy.viz requires eyepiece: pip install 'yippy[viz]'"
        ) from None
    return ep


def plot_performance_curve(
    curves_or_separations,
    values=None,
    *,
    metric="throughput",
    ax=None,
    label=None,
    line_kw=None,
):
    """Draw one metric from a record or precomputed curve arrays.

    Args:
        curves_or_separations: PerformanceCurves or separations in lambda/D.
        values: Curve values when passing bare separations; omit for a record.
        metric: Throughput, core_area_lod2, leakage_fraction,
            leakage_density_lod2, raw_contrast, or occulter_transmission.
            The last is array-only because it is not part of PerformanceCurves.
        ax: Optional single destination axes; None creates a figure and axes.
        label: Optional series label for caller-managed legends or labels.
        line_kw: Properties routed to the eyepiece line primitive.

    Returns:
        eyepiece.plot.PlotResult with the untransformed data in artist ``line``.
        Contrast and stellar leakage use a logarithmic y axis.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {tuple(_METRICS)}; got {metric!r}")
    if isinstance(curves_or_separations, PerformanceCurves):
        if values is not None:
            raise ValueError("omit values when passing PerformanceCurves")
        if metric == "occulter_transmission":
            raise ValueError("occulter_transmission requires curve arrays")
        separations = curves_or_separations.separation_lod
        values = getattr(curves_or_separations, metric)
    else:
        separations = curves_or_separations
        if values is None:
            raise ValueError("values are required with separation arrays")
    ylabel, log = _METRICS[metric]
    ep = _eyepiece()
    result = ep.plot.plot_radial(
        separations,
        values,
        ax=ax,
        label=label,
        log=log,
        line_kw=line_kw,
        xlabel=r"Separation [$\lambda/D$]",
        ylabel=ylabel,
    )
    result.ax.set_yscale("log" if log else "linear")
    ep.style.quiet_spines(result.ax)
    return result


def plot_core_mean_intensity(
    separations_lod, intensities, *, ax=None, styles=None, line_kw=None
):
    """Draw arrays evaluated from stellar_intensity_curves records.

    Args:
        separations_lod: One-dimensional radial separations in lambda/D.
        intensities: Mapping from stellar diameter (numeric lambda/D)
            to per-pixel stellar flux fractions, one array per diameter.
        ax: Optional single destination axes; None creates a figure and axes.
        styles: Optional document-wide eyepiece SourceStyles, keyed by the
            string representation of each diameter. None assigns in mapping order.
        line_kw: Properties routed to each eyepiece line primitive. Each profile
            otherwise gets a distinct palette color and marker.

    Returns:
        eyepiece.plot.PlotResult with artist ``lines`` in mapping order. Labels
        identify the stellar diameters for caller-managed legends; the y axis
        is logarithmic. No stellar maps or photometry are computed here.
    """
    if not intensities:
        raise ValueError("intensities must contain at least one diameter profile")
    ep = _eyepiece()
    if styles is None:
        styles = ep.style.SourceStyles([str(diameter) for diameter in intensities])
    lines = []
    for diameter, profile in intensities.items():
        style = styles[str(diameter)]
        diameter_lod = float(getattr(diameter, "value", diameter))
        result = ep.plot.plot_radial(
            separations_lod,
            profile,
            ax=ax,
            label=f"Diameter = {diameter_lod:g} lambda/D",
            log=True,
            color=style["color"],
            line_kw={"marker": style["marker"], **(line_kw or {})},
            xlabel=r"Separation [$\lambda/D$]",
            ylabel="Core mean intensity [stellar flux fraction / pixel]",
        )
        ax = result.ax
        lines.append(result.artists["line"])
    ep.style.quiet_spines(ax)
    return ep.plot.PlotResult(ax=ax, artists={"lines": lines})
