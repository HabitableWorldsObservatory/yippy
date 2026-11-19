"""Detector reads and locally normalized residual images."""

from dataclasses import replace

import jax
import numpy as np
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter
from optixstuff.grids import PixelGrid

from yippy.sinc.viz import _require
from yippy.sinc.viz._common import query_xy, single_ax


def _image(value):
    """Extract a two-dimensional image at the host plotting boundary."""
    data = np.asarray(getattr(value, "intensity", value), dtype=float)
    if data.ndim != 2 or not data.size:
        raise ValueError("expected a nonempty two-dimensional image")
    return data


def _display_grid(shape, pixscale_lod, pixels):
    """Resolve a plot grid from its metadata or a declared display scale."""
    if pixels is None:
        if pixscale_lod is None or not np.isfinite(pixscale_lod) or pixscale_lod <= 0:
            raise ValueError("pass positive pixscale_lod or a PixelGrid via pixels")
        return PixelGrid.regular(shape, pixscale_lod, "pixel")
    if pixscale_lod is not None:
        raise ValueError("pass pixels or pixscale_lod, not both")
    if pixels.shape != shape:
        raise ValueError("pixels.shape must match the image shape")
    return pixels


def plot_read(
    read,
    *,
    pixscale_lod=None,
    pixels=None,
    vmin=None,
    vmax=None,
    colorbar=True,
    ax=None,
    imshow_kw=None,
    cbar_kw=None,
):
    """Draw a read's intensity on its physical pixel grid with a logarithmic norm.

    Args:
        read: ImageRead or nonempty two-dimensional array.
        pixscale_lod: Positive lambda/D per pixel, centered on zero.
        pixels: Alternative PixelGrid with a diagonal basis and the
            image's shape; preserves its possibly off-center pixel origin.
        vmin: Lower display limit of the logarithmic norm; None uses nine
            decades below ``vmax`` (or below the brightest pixel).
        vmax: Upper display limit of the logarithmic norm; None uses the data.
            Fix both when animating so frames share one scale.
        colorbar: Whether to draw the inset colorbar.
        ax: Single axes; None creates a figure.
        imshow_kw: Overrides forwarded to eyepiece's image artist.
        cbar_kw: Overrides forwarded to eyepiece's inset colorbar.

    Returns:
        PlotResult with ``image``, ``cbar`` (when drawn), and
        ``update(array_or_read)``.
        Updates preserve the initial shape, extent, logarithmic floor, and norm.

    Raises:
        ValueError: Invalid image, pixel geometry, or axes shape.

    Example::

        import numpy as np
    from optixstuff.grids import PixelGrid
        from yippy.sinc.viz import plot_read
        result = plot_read(np.ones((8, 10)), pixscale_lod=0.25)
        result.update(np.full((8, 10), 2.0))
    """
    data = _image(read)
    ep = _require.eyepiece()
    grid = _display_grid(data.shape, pixscale_lod, pixels)
    ax = single_ax(ax)
    if vmin is None:
        # A coronagraph image spans many decades; nine below the brightest
        # pixel keep the halo visible without stretching to rounding noise.
        brightest = float(np.nanmax(data)) if vmax is None else float(vmax)
        vmin = brightest * 1e-9 if brightest > 0 else None
    result = ep.plot.imshow_log(
        data,
        ax=ax,
        grid=grid,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        imshow_kw=imshow_kw,
        cbar_kw=cbar_kw,
    )
    ep.plot.label_lod(ax)

    def update(array_or_read):
        values = _image(array_or_read)
        if values.shape != data.shape:
            raise ValueError("updated image must preserve the initial shape")
        result.update(values)

    return replace(result, update=update)


def plot_residual(
    read,
    truth,
    *,
    local_peak=None,
    source_xy_lod=None,
    pixscale_lod=None,
    pixels=None,
    scale="signed",
    ax=None,
    imshow_kw=None,
    cbar_kw=None,
):
    """Draw read error normalized by a fixed or truth-derived local peak.

    Args:
        read: ImageRead or two-dimensional read array.
        truth: ImageRead or truth array with the same shape.
        local_peak: Positive explicit normalizer. None uses the truth maximum
            inside radius one lambda/D of the source, or globally without one.
        source_xy_lod: Optional source pair; the radius-one boundary is inclusive.
        pixscale_lod: Positive centered pixel scale in lambda/D.
        pixels: Alternative PixelGrid with a diagonal basis.
        scale: ``"signed"`` draws the signed error on a symmetric diverging
            scale; ``"magnitude"`` draws its absolute value on a logarithmic
            scale spanning four decades below the maximum, which shows the
            spatial structure of errors far below the maximum.
        ax: Single axes; None creates a figure.
        imshow_kw: Overrides for the image artist.
        cbar_kw: Overrides for the inset colorbar.

    Returns:
        PlotResult with ``image``, ``cbar``, ``text`` (the maximum error readout),
        and ``ellipse`` (the source circle) when a source is initially supplied.
        The exact current maximum is ``abs(result.artists['image'].get_array()).max()``.
        ``update(read, truth=None, source_xy_lod=None)`` retains omitted truth
        and source and recomputes the normalizer unless local_peak was explicit.
        The initial norm, shape, extent, and artists stay fixed. A preallocated
        circle is exposed under ``ellipse`` when a source is first supplied.

    Raises:
        ValueError: Invalid images, geometry, peak, axes shape, or no pixel
            centers within the requested source disk.

    Example::

        import numpy as np
    from optixstuff.grids import PixelGrid
        from yippy.sinc.viz import plot_residual
        truth = np.ones((8, 10))
        result = plot_residual(truth + 1e-6, truth, pixscale_lod=0.25)
        result.update(truth + 2e-6)
    """
    if scale not in ("signed", "magnitude"):
        raise ValueError("scale must be 'signed' or 'magnitude'")
    data, truth_data = _image(read), _image(truth)
    source = None if source_xy_lod is None else query_xy(source_xy_lod)
    ep = _require.eyepiece()
    grid = _display_grid(data.shape, pixscale_lod, pixels)
    # Keep source-disk membership in the static grid's host precision.
    with jax.enable_x64():
        x, y = map(np.asarray, grid.coordinates())

    def resolve(values, reference, point):
        if values.shape != data.shape or reference.shape != data.shape:
            raise ValueError("read and truth must preserve the initial image shape")
        if local_peak is not None:
            peak = float(np.asarray(local_peak))
        elif point is None:
            peak = float(np.max(reference))
        else:
            inside = (x - point[0]) ** 2 + (y - point[1]) ** 2 <= 1.0
            if not np.any(inside):
                raise ValueError("no pixel centers within radius 1 lambda/D of source")
            peak = float(np.max(reference[inside]))
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError("local_peak must be finite and positive")
        residual = (values - reference) / peak
        maximum = float(np.max(np.abs(residual)))
        return residual, maximum

    residual, maximum = resolve(data, truth_data, source)
    ax = single_ax(ax)
    top = maximum if maximum > 0 else 1.0
    if scale == "signed":
        result = ep.plot.imshow_diverging(
            residual,
            ax=ax,
            grid=grid,
            vlim=top,
            imshow_kw=imshow_kw,
            cbar_kw=cbar_kw,
        )
    else:
        result = ep.plot.imshow_log(
            np.abs(residual),
            ax=ax,
            grid=grid,
            vmin=top * 1e-4,
            vmax=top,
            imshow_kw=imshow_kw,
            cbar_kw=cbar_kw,
        )
    ep.plot.label_lod(ax)
    cbar = result.artists.get("cbar")
    if cbar is not None and scale == "signed":
        # Plain exponent tick labels avoid a colorbar offset label that would
        # collide with the axes title.
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(_exponent_label))
    text = ax.text(
        0.03,
        0.03,
        f"max |error| / peak = {maximum:.1e}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize="small",
        bbox=ep.style.backing(),
    )
    artists = {**result.artists, "text": text}
    circle = Circle(
        tuple(source) if source is not None else (np.nan, np.nan),
        1.0,
        fill=False,
        edgecolor=ep.style.neutral(0.45),
        visible=source is not None,
    )
    ax.add_patch(circle)
    if source is not None:
        artists["ellipse"] = circle

    def update(read, truth=None, source_xy_lod=None):
        nonlocal truth_data, source
        values = _image(read)
        reference = truth_data if truth is None else _image(truth)
        point = source if source_xy_lod is None else query_xy(source_xy_lod)
        residual, maximum = resolve(values, reference, point)
        result.update(residual if scale == "signed" else np.abs(residual))
        text.set_text(f"max |error| / peak = {maximum:.1e}")
        if point is not None:
            circle.center = tuple(point)
            circle.set_visible(True)
            artists["ellipse"] = circle
        truth_data, source = reference, point

    return replace(result, artists=artists, update=update)


def _exponent_label(value, _position):
    """Format a colorbar tick as a compact signed exponent label."""
    if value == 0:
        return "0"
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")
