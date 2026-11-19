"""Numerical and composition contracts for the plotting boundary."""

# The optional dependency guard and Agg selection must precede plotting imports.
# ruff: noqa: E402
import pytest
from optixstuff.grids import Frame, PixelGrid

ep = pytest.importorskip("eyepiece")

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import yippy.sinc as sc
from yippy.sinc import viz
from yippy.sinc.kernel import KernelProfile, stencil

# tol_colors builds its palettes with Matplotlib's pending-deprecated set_bad;
# eyepiece's suite uses the same narrow upstream compatibility filter.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The set_bad function will be deprecated.*"
    ":PendingDeprecationWarning:tol_colors"
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _call(name, ax=None):
    image = np.arange(1.0, 49.0).reshape(6, 8)
    calls = {
        "plot_kernel": lambda: viz.plot_kernel(ax=ax),
        "plot_stencil": lambda: viz.plot_stencil(0.3, profile=KernelProfile(), ax=ax),
        "plot_stencil_2d": lambda: viz.plot_stencil_2d(
            (0.3, 0.4), ((0, 0), (0.5, 0.5), (40, 40)), profile=KernelProfile(), ax=ax
        ),
        "plot_spacing_rule": lambda: viz.plot_spacing_rule(ax=ax),
        "plot_coverage": lambda: viz.plot_coverage(
            ((-5.0, -6.0), (0.25, 0.5), (40, 48)), query=(0.0, 2.0), ax=ax
        ),
        "plot_read": lambda: viz.plot_read(image, pixscale_lod=0.25, ax=ax),
        "plot_residual": lambda: viz.plot_residual(
            image + 0.01, image, pixscale_lod=0.25, ax=ax
        ),
        "plot_check_residuals": lambda: viz.plot_check_residuals(
            independent=[0.0, 1e-6, np.nan], mirror=[1e-7], axis=[0.0], ax=ax
        ),
        "plot_band_spectrum": lambda: viz.plot_band_spectrum(
            np.cos(np.arange(64.0)), 0.25, bandwidth=1.0, ax=ax
        ),
    }
    return calls[name]()


NAMES = (
    "plot_kernel",
    "plot_stencil",
    "plot_stencil_2d",
    "plot_spacing_rule",
    "plot_coverage",
    "plot_read",
    "plot_residual",
    "plot_check_residuals",
    "plot_band_spectrum",
)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("handed", [False, True])
def test_result_and_equal_slots(name, handed):
    ax = plt.subplots(1, 2)[1][0] if handed else None
    result = _call(name, ax)
    assert isinstance(result, ep.plot.PlotResult)
    assert result.artists
    assert set(result.artists) <= ep.plot.ARTIST_KEYS
    if handed:
        assert result.ax is ax
        result.fig.canvas.draw()
        widths = [a.get_position(original=True).width for a in result.fig.axes]
        np.testing.assert_allclose(widths, [widths[0], widths[0]])


@pytest.mark.parametrize("name", NAMES)
def test_rejects_axes_array(name):
    axes = plt.subplots(1, 2)[1]
    with pytest.raises(ValueError, match=r"single ax.*\(2,\)"):
        _call(name, axes)


def test_kernel_support_and_profile():
    result = viz.plot_kernel(sc.KernelProfile())
    line = result.artists["line"]
    assert len(line.get_xdata()) == 2001
    np.testing.assert_allclose(line.get_xdata()[[0, -1]], [-17, 17])
    assert np.all(line.get_ydata()[np.abs(line.get_xdata()) >= 16] == 0)
    assert len(result.artists["lines"]) == 3
    assert len(viz.plot_kernel(show_ideal=False).artists["lines"]) == 2
    with pytest.raises((ValueError, TypeError)):
        viz.plot_kernel("invalid")


def test_routed_color_aliases_override_defaults():
    kernel = viz.plot_kernel(line_kw={"c": "red"})
    assert kernel.artists["line"].get_color() == "red"
    spectrum = viz.plot_band_spectrum(
        np.ones(8), 0.25, bandwidth=1, line_kw={"c": "red"}
    )
    assert spectrum.artists["line"].get_color() == "red"
    coverage = viz.plot_coverage(((0, 0), (1, 1), (2, 2)), node_kw={"c": np.arange(4)})
    np.testing.assert_array_equal(
        coverage.artists["scatter"][0].get_array(), np.arange(4)
    )


def test_stencil_known_answer_and_update():
    result = viz.plot_stencil(
        0.3,
        profile=KernelProfile(),
        origin=-2.0,
        spacing=0.25,
        n_nodes=40,
        samples=np.arange(40.0),
    )
    stems = result.artists["collection"]
    artists = list(result.ax.get_children())
    for query in (0.3, -2.0, 6.2):
        result.update(query)
        taps, weights = stencil(jnp.asarray([query]), -2.0, 0.25)
        segments = np.asarray(stems.get_segments())
        np.testing.assert_allclose(segments[:, 1, 0], -2 + np.asarray(taps[0]) * 0.25)
        np.testing.assert_allclose(segments[:, 1, 1], weights[0], atol=1e-15)
        np.testing.assert_allclose(np.sum(segments[:, 1, 1]), 1.0, atol=1e-5)
        fresh = viz.plot_stencil(
            query,
            profile=KernelProfile(),
            origin=-2.0,
            spacing=0.25,
            n_nodes=40,
            samples=np.arange(40.0),
        )
        np.testing.assert_allclose(segments, fresh.artists["collection"].get_segments())
        for current, expected in zip(
            result.artists["scatter"], fresh.artists["scatter"], strict=True
        ):
            np.testing.assert_allclose(current.get_offsets(), expected.get_offsets())
            np.testing.assert_allclose(
                current.get_facecolors(), expected.get_facecolors()
            )
        if query == -2.0:
            expected = np.zeros(32)
            expected[15] = 1.0
            np.testing.assert_allclose(segments[:, 1, 1], expected, atol=1e-12)
    assert artists == list(result.ax.get_children())


def test_stencil_requires_x64():
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="jax_enable_x64"):
            viz.plot_stencil(0.0, profile=KernelProfile())


def test_stencil_callable_samples():
    result = viz.plot_stencil(2.5, profile=KernelProfile(), samples=lambda x: x**2)
    points = result.artists["scatter"][-1].get_offsets()
    np.testing.assert_allclose(points[:, 1], points[:, 0] ** 2)


@pytest.mark.parametrize(
    "quadrant, origin, expected",
    [
        (False, (-5.0, -6.0), (-1.25, 1.5, 4.25, 4.5)),
        (True, (0.0, 0.0), (-8.0, -12.0, 16.0, 24.0)),
    ],
)
def test_coverage_geometry_and_update(quadrant, origin, expected):
    lattice = sc.source_grid(
        jnp.array(origin),
        jnp.array([0.25, 0.5]),
        (40, 48),
        Frame(500.0, 6.0, "circumscribed"),
    )
    domain = sc.Domain(
        "box", "closed", jnp.array([[-1.0, -2.0], [1.0, 2.0]]), jnp.zeros(2)
    )
    result = viz.plot_coverage(
        lattice, quadrant=quadrant, query=(0.0, 2.0), domain=domain
    )
    served = result.artists["fill"]
    np.testing.assert_allclose(
        [*served.get_xy(), served.get_width(), served.get_height()], expected
    )
    if quadrant:
        stored, virtual, _ = result.artists["scatter"]
        points = np.concatenate([stored.get_offsets(), virtual.get_offsets()])
        assert len(points) == 95 * 79
        assert len(np.unique(points, axis=0)) == len(points)
    artists = list(result.ax.get_children())
    result.update((0.3, -0.2))
    fresh = viz.plot_coverage(
        lattice, quadrant=quadrant, query=(0.3, -0.2), domain=domain
    )
    np.testing.assert_allclose(
        result.artists["scatter"][-1].get_offsets(),
        fresh.artists["scatter"][-1].get_offsets(),
    )
    rectangle = result.artists["ellipse"][-1]
    np.testing.assert_allclose(
        rectangle.get_xy(), fresh.artists["ellipse"][-1].get_xy()
    )
    assert rectangle.get_width() == 8.0
    assert rectangle.get_height() == 16.0
    assert artists == list(result.ax.get_children())


def test_coverage_annulus():
    domain = sc.Domain("annulus", "closed", jnp.zeros((2, 2)), jnp.array([1.0, 2.0]))
    result = viz.plot_coverage(((0, 0), (0.25, 0.25), (40, 40)), domain=domain)
    assert [p.radius for p in result.artists["ellipse"]] == [1.0, 2.0]


def test_updates_can_introduce_query_and_source_without_new_artists():
    lattice = ((0, 0), (0.25, 0.25), (40, 40))
    coverage = viz.plot_coverage(lattice)
    children = list(coverage.ax.get_children())
    coverage.update((0.3, 0.4))
    fresh = viz.plot_coverage(lattice, query=(0.3, 0.4))
    np.testing.assert_allclose(
        coverage.artists["scatter"][-1].get_offsets(),
        fresh.artists["scatter"][-1].get_offsets(),
    )
    np.testing.assert_allclose(
        coverage.artists["ellipse"][-1].get_xy(), fresh.artists["ellipse"][-1].get_xy()
    )
    assert children == list(coverage.ax.get_children())


def test_update_can_introduce_source_without_new_artists():
    truth = np.arange(1.0, 36.0).reshape(5, 7)
    residual = viz.plot_residual(truth + 1, truth, pixscale_lod=0.5)
    children = list(residual.ax.get_children())
    residual.update(truth + 1, source_xy_lod=(0, 0))
    fresh = viz.plot_residual(truth + 1, truth, pixscale_lod=0.5, source_xy_lod=(0, 0))
    np.testing.assert_allclose(
        residual.artists["image"].get_array(), fresh.artists["image"].get_array()
    )
    assert residual.artists["ellipse"].center == (0.0, 0.0)
    assert children == list(residual.ax.get_children())


def test_pixel_centers_and_read_update(full_reader):
    read = full_reader.read(jnp.array([0.0, 0.0]), jnp.array(0, dtype=jnp.int32))
    pixels = full_reader.products.pixels
    result = viz.plot_read(read, pixels=pixels)
    result.update(np.ones(pixels.shape) * 2)
    fresh = viz.plot_read(np.ones(pixels.shape) * 2, pixels=pixels)
    np.testing.assert_allclose(
        result.artists["image"].get_array(), fresh.artists["image"].get_array()
    )
    custom = PixelGrid(
        (3, 4),
        (jnp.array([1.0, 0.0]))[0],
        (jnp.array([1.0, 0.0]))[1],
        jnp.diag(jnp.array([0.5, 0.25])),
        "lod",
        Frame(500.0, 6.0, "circumscribed"),
    )
    image = viz.plot_read(np.ones((3, 4)), pixels=custom).artists["image"]
    np.testing.assert_allclose(image.get_extent(), [-0.75, 1.25, -0.125, 0.625])


def test_residual_local_peak_and_update():
    truth = np.ones((5, 7))
    truth[2, 4] = 5.0  # exactly one lambda/D from the source
    truth[0, 0] = 100.0
    result = viz.plot_residual(truth + 1, truth, source_xy_lod=(0, 0), pixscale_lod=1)
    np.testing.assert_allclose(result.artists["image"].get_array(), 0.2)
    assert result.artists["text"].get_text() == "max |error| / peak = 2.0e-01"
    children = list(result.ax.get_children())
    result.update(truth + 2, truth=truth * 2, source_xy_lod=(1, 0))
    fresh = viz.plot_residual(
        truth + 2, truth * 2, source_xy_lod=(1, 0), pixscale_lod=1
    )
    np.testing.assert_allclose(
        result.artists["image"].get_array(), fresh.artists["image"].get_array()
    )
    assert result.artists["text"].get_text() == fresh.artists["text"].get_text()
    assert result.artists["ellipse"].center == (1.0, 0.0)
    assert children == list(result.ax.get_children())


def test_residual_local_peak_excludes_pixel_just_outside_source_disk():
    """Fails if float32 rounding includes a sample outside the source disk."""
    pixels = PixelGrid.regular(
        (1, 2), 1, "lod", center=(-1e-8, 0), frame=Frame(500, 6, "circumscribed")
    )
    truth = np.array([[2.0, 100.0]])
    with jax.enable_x64(False):
        result = viz.plot_residual(
            truth + 1, truth, pixels=pixels, source_xy_lod=(0, 0)
        )
    np.testing.assert_allclose(result.artists["image"].get_array(), 0.5)
    plt.close(result.fig)


def test_residual_global_and_fixed_peak():
    truth = np.arange(1.0, 13.0).reshape(3, 4)
    for peak in (None, 2.0):
        result = viz.plot_residual(truth + 1, truth, local_peak=peak, pixscale_lod=0.5)
        expected = 1 / (12 if peak is None else 2)
        np.testing.assert_allclose(result.artists["image"].get_array(), expected)
        result.update(truth + 2)
        np.testing.assert_allclose(result.artists["image"].get_array(), expected * 2)


def test_residual_magnitude_scale_and_update():
    truth = np.ones((4, 4))
    signed = truth.copy()
    signed[1, 2] -= 1e-3
    signed[3, 0] += 1e-6
    result = viz.plot_residual(signed, truth, scale="magnitude", pixscale_lod=0.5)
    shown = np.asarray(result.artists["image"].get_array())
    expected = np.abs(signed - truth)
    nonzero = expected > 0
    np.testing.assert_allclose(shown[nonzero], expected[nonzero])
    norm = result.artists["image"].norm
    np.testing.assert_allclose((norm.vmin, norm.vmax), (1e-7, 1e-3))
    result.update(truth - 2e-3)
    np.testing.assert_allclose(result.artists["image"].get_array(), 2e-3, rtol=1e-9)
    with pytest.raises(ValueError, match="scale"):
        viz.plot_residual(signed, truth, scale="log", pixscale_lod=0.5)


def test_read_display_limits_and_colorbar_switch():
    image = np.full((4, 4), 1e-3)
    image[0, 0] = 1.0
    default = viz.plot_read(image, pixscale_lod=0.25)
    np.testing.assert_allclose(default.artists["image"].norm.vmin, 1e-9)
    fixed = viz.plot_read(image, pixscale_lod=0.25, vmin=1e-6, vmax=0.5, colorbar=False)
    norm = fixed.artists["image"].norm
    np.testing.assert_allclose((norm.vmin, norm.vmax), (1e-6, 0.5))
    assert "cbar" not in fixed.artists


def test_receipt_zeros_and_nan_counts():
    receipt = sc.Receipt(
        empirical_local_peak=(0.0, 0.0, np.nan, 2e-5),
        mirror_local_peak=(0.0, 1e-8),
        axis_relative=(np.nan, 3e-7, 0.0),
        axis_exact=(True, False, True),
    )
    result = viz.plot_check_residuals(receipt)
    labels = [t.get_text() for t in result.ax.get_yticklabels()]
    assert "2 exactly 0" in labels[0] and "1 NaN" in labels[0]
    assert "1 exactly 0" in labels[1]
    assert "2 exactly 0" in labels[2] and "NaN" not in labels[2]
    assert result.ax.get_xscale() == "log"
    assert [len(p.get_offsets()) for p in result.artists["scatter"]] == [
        1,
        2,
        1,
        1,
        1,
        2,
    ]
    override = viz.plot_check_residuals(receipt, independent=[1e-7])
    assert "exactly 0" not in override.ax.get_yticklabels()[0].get_text()


def test_spectrum_known_frequency_and_update():
    x = np.arange(128) * 0.25
    result = viz.plot_band_spectrum(np.cos(2 * np.pi * 0.5 * x), 0.25, bandwidth=1)
    line = result.artists["line"]
    freq, power = line.get_data()
    np.testing.assert_allclose(abs(freq[np.argmax(power)]), 0.5)
    assert np.max(power) == 1
    assert np.min(power) >= 1e-12
    children = list(result.ax.get_children())
    trace = np.cos(2 * np.pi * 0.75 * x)
    result.update(trace)
    fresh = viz.plot_band_spectrum(trace, 0.25, bandwidth=1)
    np.testing.assert_allclose(line.get_data(), fresh.artists["line"].get_data())
    assert children == list(result.ax.get_children())


def _taps_2d(result):
    """Return offsets and areas of the counted taps."""
    taps = result.artists["scatter"][-2]
    return np.asarray(taps.get_offsets()), np.asarray(taps.get_sizes())


def test_stencil_2d_areas_are_the_separable_weights():
    lattice = ((0.0, 0.0), (0.25, 0.5), (60, 40))
    query = (3.3, 6.1)
    result = viz.plot_stencil_2d(
        query, lattice, profile=KernelProfile(), max_area=100.0
    )
    offsets, areas = _taps_2d(result)
    assert len(offsets) == 1024
    tx, wx = stencil(jnp.asarray([query[0]]), 0.0, 0.25)
    ty, wy = stencil(jnp.asarray([query[1]]), 0.0, 0.5)
    expected = np.outer(np.asarray(wy[0]), np.asarray(wx[0])).ravel()
    np.testing.assert_allclose(areas, 100.0 * np.abs(expected))
    np.testing.assert_allclose(np.sum(expected), 1.0, atol=1e-5)
    np.testing.assert_allclose(offsets[:32, 0], 0.25 * np.asarray(tx[0]))
    np.testing.assert_allclose(offsets[::32, 1], 0.5 * np.asarray(ty[0]))
    colors = np.asarray(result.artists["scatter"][-2].get_edgecolors())
    assert np.all(colors[expected >= 0] == colors[np.argmax(expected)])
    assert len(np.unique(colors, axis=0)) == 2


def test_stencil_2d_weight_order_counts_largest_first():
    lattice = ((0.0, 0.0), (0.25, 0.25), (60, 60))
    result = viz.plot_stencil_2d(
        (3.3, 4.1),
        lattice,
        profile=KernelProfile(),
        order="weight",
        n_taps=0,
        max_area=1.0,
    )
    assert len(_taps_2d(result)[0]) == 0
    result.update(n_taps=10)
    top = _taps_2d(result)[1]
    result.update(n_taps=1024)
    everything = np.sort(_taps_2d(result)[1])[::-1]
    np.testing.assert_allclose(np.sort(top)[::-1], everything[:10])
    result.update(n_taps=11)
    assert len(_taps_2d(result)[0]) == 11
    with pytest.raises(ValueError, match="n_taps"):
        result.update(n_taps=1025)


def test_stencil_2d_quadrant_fold_and_update():
    lattice = ((0.0, 0.0), (0.5, 0.5), (31, 31))
    result = viz.plot_stencil_2d(
        (0.7, 4.0), lattice, profile=KernelProfile(), quadrant=True
    )
    signed, _ = _taps_2d(result)
    assert signed[:, 0].min() < 0
    face = np.asarray(result.artists["scatter"][-2].get_facecolors())
    hollow = face[:, 3] == 0.0
    assert hollow.sum() == 0  # every mirrored tap is stored
    result.update(fold=1.0)
    folded, _ = _taps_2d(result)
    np.testing.assert_allclose(folded[:, 0], np.abs(signed[:, 0]))
    np.testing.assert_allclose(folded[:, 1], np.abs(signed[:, 1]))
    result.update((-0.7, 4.0), fold=0.0)
    fresh = viz.plot_stencil_2d(
        (-0.7, 4.0), lattice, profile=KernelProfile(), quadrant=True
    )
    np.testing.assert_allclose(_taps_2d(result)[0], _taps_2d(fresh)[0])
    with pytest.raises(ValueError, match="fold"):
        viz.plot_stencil_2d(
            (0.0, 0.0),
            ((-5, -5), (1, 1), (40, 40)),
            profile=KernelProfile(),
            fold=0.5,
        )
    with pytest.raises(ValueError, match="quadrant"):
        viz.plot_stencil_2d(
            (0.0, 0.0),
            ((-5, -5), (1, 1), (40, 40)),
            profile=KernelProfile(),
            quadrant=True,
        )


def test_stencil_2d_marks_taps_outside_storage_hollow():
    result = viz.plot_stencil_2d(
        (1.0, 1.0), ((0.0, 0.0), (0.5, 0.5), (40, 40)), profile=KernelProfile()
    )
    face = np.asarray(result.artists["scatter"][-2].get_facecolors())
    offsets, _ = _taps_2d(result)
    outside = (offsets < 0).any(axis=1)
    assert outside.any()
    assert np.all(face[outside, 3] == 0.0)
    assert np.all(face[~outside, 3] > 0.0)


def test_spacing_rule_chain_of_numbers():
    result = viz.plot_spacing_rule(support_width=1.0, radius=(4.18, 5.70))
    texts = [t.get_text() for t in result.artists["text"]]
    assert "27" in texts[3] and "11+16" in texts[3].replace(" ", "")
    assert "27" in texts[4] and "31" in texts[4]
    assert "0.38" in texts[2]
    wider = viz.plot_spacing_rule(support_width=1.2, sigma_max=1.0, radius=4.0)
    labels = [t.get_text() for t in wider.artists["text"]]
    assert "0.317" in labels[2]
    assert "1.2" in labels[1]


def test_spacing_rule_reads_products_and_lattice(quadrant_delivery):
    base = quadrant_delivery["products"]
    products = eqx.tree_at(lambda p: p.sampling_min_nm, base, jnp.array([450.0]))
    result = viz.plot_spacing_rule(products, radius=3.0)
    texts = [t.get_text() for t in result.artists["text"]]
    assert "1.11" in texts[1]
    lattice = ((0.0, 0.0), (0.3, 0.3), (27, 31))
    given = viz.plot_spacing_rule(products, lattice)
    assert "27" in given.artists["text"][4].get_text()
    with pytest.raises(ValueError, match="positive"):
        viz.plot_spacing_rule(support_width=-1.0)
    with pytest.raises(ValueError, match="margin"):
        viz.plot_spacing_rule(margin=1.5)


def test_stencil_viz_draws_the_reader_profile(full_reader, tone_full):
    prefiltered = tone_full["reader"]
    """The drawn taps follow the reader's own profile, not a fixed tap count."""
    query = (0.31, -0.17)
    for reader in (full_reader, prefiltered):
        found = reader.stencil(jnp.asarray(query))
        count = reader.profile.taps
        result = viz.plot_stencil_2d(query, reader, max_area=100.0)
        offsets, areas = _taps_2d(result)
        assert len(offsets) == count**2
        expected = np.outer(np.asarray(found.weights[1]), np.asarray(found.weights[0]))
        np.testing.assert_allclose(areas, 100.0 * np.abs(expected).ravel())
        line = viz.plot_stencil(query[0], reader=reader, axis=0)
        segments = np.asarray(line.artists["collection"].get_segments())
        assert len(segments) == count
        np.testing.assert_allclose(segments[:, 1, 1], found.weights[0], atol=1e-15)
    assert prefiltered.profile.taps != full_reader.profile.taps


def test_stencil_viz_source_arguments(tone_full):
    lattice = ((0.0, 0.0), (0.5, 0.5), (40, 40))
    with pytest.raises(ValueError, match="needs a profile"):
        viz.plot_stencil_2d((0.0, 0.0), lattice)
    with pytest.raises(ValueError, match="supplies its own"):
        viz.plot_stencil_2d((0.0, 0.0), tone_full["reader"], profile=KernelProfile())
    with pytest.raises(ValueError, match="needs the beta and scale"):
        viz.plot_stencil_2d((0.0, 0.0), lattice, profile=sc.PrefilteredKernelProfile())
    with pytest.raises(ValueError, match="exactly one"):
        viz.plot_stencil(0.0)
