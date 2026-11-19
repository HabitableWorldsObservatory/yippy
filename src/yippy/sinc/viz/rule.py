"""The spacing rule as five small panels: from support width to a starter grid."""

import math

import numpy as np
from matplotlib.patches import Circle, Rectangle

from yippy.sinc.viz import _require
from yippy.sinc.viz._common import lattice_parts, single_ax
from yippy.sinc.viz.coverage import plot_coverage

_STENCIL_MARGIN = 16


def _physical_inputs(products, support_width, sigma_max):
    """Resolve support width and inverse wavelength from Products or numbers."""
    if products is None:
        return (
            1.0 if support_width is None else float(support_width),
            1.0 if sigma_max is None else float(sigma_max),
        )
    reference = float(products.frame.reference_wavelength_nm)
    shortest = [float(v) for v in np.asarray(products.sampling_min_nm)]
    campaign = float(products.campaign_sampling_min_nm)
    if np.isfinite(campaign):
        shortest.append(campaign)
    width = float(np.max(np.asarray(products.support_extent_xy)[:, 0]))
    return (
        width if support_width is None else float(support_width),
        reference / min(shortest) if sigma_max is None else float(sigma_max),
    )


def plot_spacing_rule(
    products=None,
    lattice=None,
    *,
    support_width=None,
    sigma_max=None,
    radius=5.0,
    margin=0.76,
    ax=None,
):
    """Chain the five quantities that fix a delivery: L, B, h, node count, grid.

    The panels run left to right: the illuminated support width ``L``, the
    bandwidth ``B = sigma_max * L``, the largest admissible spacing
    ``h = margin / (2 B)`` beside the Nyquist spacing, the node count per axis
    ``ceil(R / h) + 16`` and the resulting quadrant lattice with the region its
    complete stencils serve. Everything is drawn on the x axis; the lattice
    panel shows both.

    Args:
        products: Optional Products. Its widest x support extent and its
            shortest sampling wavelength (and the campaign minimum, when
            finite) set ``L`` and ``sigma_max``.
        lattice: Optional PixelGrid or ``(origin_xy, spacing_xy, shape_yx)``
            tuple of a real delivery. Its x spacing replaces the rule's ``h``
            in the spacing panel, and its served half-width replaces
            ``radius``. Without it the rule builds the square lattice itself.
        support_width: Full illuminated width in units of D; overrides the
            value read from products. Default 1 when no products are given.
        sigma_max: Largest ratio of the reference to a contributing
            wavelength; overrides the products. Default 1.
        radius: Served half-width ``R`` in lambda_0/D, a number or an
            ``(R_x, R_y)`` pair. Ignored when a lattice is given.
        margin: Largest oversampling factor the kernel admits.
        ax: Single axes, turned into a strip of five inset panels.

    Returns:
        PlotResult with ``text`` holding the five value labels in panel order,
        ``scatter`` the nodes of the spacing panel, ``fill`` the bandwidth
        band and ``ellipse`` the pupil. A strip has no ``update``.

    Raises:
        ValueError: Non-positive inputs, an invalid lattice or axes shape.

    Example::

        from yippy.sinc.viz import plot_spacing_rule
        result = plot_spacing_rule(support_width=1.0, radius=(4.18, 5.70))
    """
    width, sigma = _physical_inputs(products, support_width, sigma_max)
    if not (np.isfinite(width) and width > 0 and np.isfinite(sigma) and sigma > 0):
        raise ValueError("support width and sigma_max must be positive and finite")
    if not 0 < margin <= 1:
        raise ValueError("margin must lie in (0, 1]")
    bandwidth = sigma * width
    rule_spacing = margin / (2 * bandwidth)
    if lattice is None:
        reach = np.broadcast_to(np.asarray(radius, dtype=float), (2,))
        if not np.all(np.isfinite(reach)) or np.any(reach <= 0):
            raise ValueError("radius must be positive and finite")
        nodes_xy = [math.ceil(r / rule_spacing) + _STENCIL_MARGIN for r in reach]
        lattice = ((0.0, 0.0), (rule_spacing, rule_spacing), (nodes_xy[1], nodes_xy[0]))
    _, spacing, shape = lattice_parts(lattice)
    nodes_x = int(shape[1])
    spacing_x = float(spacing[0])
    reach_x = (nodes_x - _STENCIL_MARGIN) * spacing_x
    to_reach = nodes_x - _STENCIL_MARGIN
    ep = _require.eyepiece()
    ax = single_ax(ax)
    ax.set_axis_off()
    ink, data = ep.style.role("text"), ep.style.role("data")
    scenery = ep.style.neutral(0.45)
    insets = [
        ax.inset_axes([0.2 * i + 0.012, 0.3, 0.176, 0.56], zorder=2) for i in range(5)
    ]
    titles = (
        "support width",
        "bandwidth",
        "spacing",
        "nodes per axis",
        "quadrant grid",
    )
    labels = (
        rf"$L={width:.3g}\,D$",
        rf"$B=\sigma_{{\max}}L={bandwidth:.3g}$",
        rf"$h={spacing_x:.3g}$",
        rf"${to_reach}+{_STENCIL_MARGIN}={nodes_x}$",
        rf"${nodes_x}\times{int(shape[0])}$",
    )
    for panel, title in zip(insets, titles, strict=True):
        panel.set_title(title, fontsize=9, color=ink)
    texts = []

    pupil_panel = insets[0]
    pupil = Circle((0, 0), width / 2, fill=False, edgecolor=data, linewidth=1.6)
    pupil_panel.add_patch(pupil)
    pupil_panel.annotate(
        "",
        (-width / 2, -0.7 * width),
        (width / 2, -0.7 * width),
        arrowprops={"arrowstyle": "<->", "color": scenery},
    )
    pupil_panel.set_xlim(-0.8 * width, 0.8 * width)
    pupil_panel.set_ylim(-0.9 * width, 0.6 * width)
    pupil_panel.set_aspect("equal")

    band_panel = insets[1]
    edge = 1.7 * bandwidth
    band = band_panel.fill(
        [-bandwidth, 0, bandwidth], [0, 1, 0], color=data, alpha=0.55, linewidth=0
    )[0]
    band_panel.axhline(0, color=scenery, linewidth=0.8)
    band_panel.set_xlim(-edge, edge)
    band_panel.set_ylim(-0.12, 1.2)
    band_panel.set_xticks(
        [-bandwidth, 0, bandwidth], [f"{-bandwidth:.3g}", "0", f"{bandwidth:.3g}"]
    )
    band_panel.set_yticks([])
    band_panel.set_xlabel("cycles per $\\lambda_0/D$", fontsize=8)

    spacing_panel = insets[2]
    nyquist = 1 / (2 * bandwidth)
    dots = []
    for level, step, color in ((1, nyquist, scenery), (0, spacing_x, data)):
        positions = np.arange(6) * step
        dots.append(
            spacing_panel.scatter(positions, np.full(6, level), s=14, color=color)
        )
    spacing_panel.text(0, 1.45, f"Nyquist {nyquist:.3g}", color=scenery, fontsize=8)
    spacing_panel.text(
        0, -0.75, f"$\\rho={2 * bandwidth * spacing_x:.2f}$", color=data, fontsize=8
    )
    spacing_panel.set_xlim(-0.1 * nyquist, 5.6 * nyquist)
    spacing_panel.set_ylim(-1.0, 1.9)
    spacing_panel.set_yticks([])
    spacing_panel.set_xlabel("source position ($\\lambda_0/D$)", fontsize=8)

    count_panel = insets[3]
    count_panel.add_patch(Rectangle((0, 0), to_reach, 1, facecolor=data, alpha=0.8))
    count_panel.add_patch(
        Rectangle((to_reach, 0), _STENCIL_MARGIN, 1, facecolor=scenery, alpha=0.8)
    )
    count_panel.text(
        to_reach / 2, 1.25, f"$R/h$: {to_reach}", ha="center", color=data, fontsize=8
    )
    count_panel.text(
        to_reach + _STENCIL_MARGIN / 2,
        -0.6,
        "stencil\nmargin",
        ha="center",
        color=scenery,
        fontsize=8,
    )
    count_panel.set_xlim(-1, nodes_x + 1)
    count_panel.set_ylim(-1.5, 2.2)
    count_panel.set_yticks([])
    count_panel.set_xlabel("node index", fontsize=8)

    grid_panel = insets[4]
    grid_result = plot_coverage(
        lattice, quadrant=True, ax=grid_panel, node_kw={"s": 2.5, "linewidths": 0}
    )
    # The mirrored copies are implied by symmetry and not stored: hide them so
    # the stored quadrant and the stencil box read as two distinct things.
    for virtual in grid_result.artists["scatter"][1:]:
        virtual.set_visible(False)
    extent_x = (nodes_x - 1) * spacing_x
    extent_y = (int(shape[0]) - 1) * float(spacing[1])
    reach_y = (int(shape[0]) - _STENCIL_MARGIN) * float(spacing[1])
    pad_x, pad_y = 1.08 * extent_x, 1.08 * extent_y
    grid_panel.set_xlim(-pad_x * 0.45, pad_x)
    grid_panel.set_ylim(-pad_y * 0.45, pad_y)
    grid_panel.axhline(0, color=scenery, linewidth=0.5)
    grid_panel.axvline(0, color=scenery, linewidth=0.5)
    grid_panel.set_xlabel("")
    grid_panel.set_ylabel("")
    grid_panel.set_xticks([])
    grid_panel.set_yticks([])
    grid_panel.text(
        0.97,
        0.04,
        "stored\nnodes",
        transform=grid_panel.transAxes,
        ha="right",
        color=data,
        fontsize=8,
    )
    grid_panel.text(
        0.03,
        0.03,
        f"stencil box\n$|s|<({reach_x:.3g},\\,{reach_y:.3g})$",
        transform=grid_panel.transAxes,
        ha="left",
        va="bottom",
        color=ink,
        fontsize=7,
    )

    for panel, label in zip(insets, labels, strict=True):
        texts.append(
            panel.text(
                0.5,
                -0.42,
                label,
                transform=panel.transAxes,
                ha="center",
                va="top",
                color=ink,
                fontsize=10,
            )
        )
    return ep.plot.PlotResult(
        ax, {"text": texts, "scatter": dots, "fill": band, "ellipse": [pupil]}
    )
