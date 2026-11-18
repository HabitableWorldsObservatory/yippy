"""Stateless performance plots; install ``yippy[viz]`` to draw them.

Style activation and figure saving belong to the caller. Importing this module
does not load eyepiece or Matplotlib or change the active style.
"""

from optixstuff.performance import PerformanceCurves, SampledProfile

__all__ = ["plot_core_mean_intensity", "plot_performance_curve"]

_METRICS = {
    "throughput": ("Throughput", False),
    "throughput_variance": ("Planet shot-noise fraction", False),
    "core_area_variance_lod2": (r"Variance area [$(\lambda/D)^2$]", False),
    "leakage_fraction_variance": ("Stellar shot-noise fraction", True),
    "leakage_density_variance_lod2": (
        r"Stellar shot-noise density [$(\lambda/D)^{-2}$]",
        True,
    ),
    "core_area_lod2": (r"Core area [$(\lambda/D)^2$]", False),
    "leakage_fraction": ("Stellar leakage fraction", True),
    "leakage_density_lod2": (r"Stellar leakage density [$(\lambda/D)^{-2}$]", True),
    "raw_contrast": ("Raw contrast", True),
    "occulter_transmission": ("Occulter transmission", False),
    "stellar_ring_mean_per_pixel": (
        "Stellar ring mean [stellar flux fraction / pixel]",
        True,
    ),
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
    metric=None,
    ax=None,
    label=None,
    line_kw=None,
):
    """Draw one quantity from a record, sampled profile or curve arrays.

    Args:
        curves_or_separations: A record, sampled profile, or separations in lambda/D.
        values: Curve values for bare separations.
        metric: Record field for throughput, collecting area, leakage fraction
            or leakage density (mean or variance), or raw_contrast. A profile
            supplies its own quantity. Otherwise the default is throughput.
        ax: Optional single destination axes; None creates a figure and axes.
        label: Optional series label for caller-managed legends or labels.
        line_kw: Properties routed to the eyepiece line primitive.

    Returns:
        eyepiece.plot.PlotResult with the untransformed data in artist ``line``.
        Contrast and stellar leakage use a logarithmic y axis.
    """
    if isinstance(curves_or_separations, SampledProfile):
        profile = curves_or_separations
        if values is not None or (metric is not None and metric != profile.quantity):
            raise ValueError("the sampled profile supplies its own quantity and values")
        metric = profile.quantity
        separations, values = profile.separation_lod, profile.values
    else:
        metric = "throughput" if metric is None else metric
        if isinstance(curves_or_separations, PerformanceCurves):
            if values is not None or metric not in curves_or_separations.interpolation:
                raise ValueError(
                    f"select a metric stored in the record; got {metric!r}"
                )
            separations = curves_or_separations.separation_lod
            values = getattr(curves_or_separations, metric)
        else:
            separations = curves_or_separations
            if values is None:
                raise ValueError("values are required with separation arrays")
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {tuple(_METRICS)}; got {metric!r}")
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


def plot_core_mean_intensity(profiles, *, ax=None, styles=None, line_kw=None):
    """Draw per-pixel ring means returned by stellar_intensity_curves.

    Args:
        profiles: Mapping from stellar diameter in lambda/D to its SampledProfile.
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
    if not profiles:
        raise ValueError("profiles must contain at least one diameter profile")
    ep = _eyepiece()
    if styles is None:
        styles = ep.style.SourceStyles([str(diameter) for diameter in profiles])
    lines = []
    for diameter, profile in profiles.items():
        if profile.quantity != "stellar_ring_mean_per_pixel":
            raise ValueError("expected a stellar_ring_mean_per_pixel profile")
        style = styles[str(diameter)]
        diameter_lod = float(getattr(diameter, "value", diameter))
        result = ep.plot.plot_radial(
            profile.separation_lod,
            profile.values,
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
