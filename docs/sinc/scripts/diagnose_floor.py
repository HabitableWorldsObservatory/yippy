"""Separate the candidate causes of the pixel-relative read error on the delivery.

The worst error of the 16 interior checks, measured against each pixel's brightest
stored value, is larger than the kernel's own floor. This script tests three
candidate causes with measurements:

* storage precision: re-propagated float64 nodes against the delivered float32,
  and the reads from each;
* a band wider than declared: the spectrum of a pixel's response along each
  source axis, from a fine line of direct propagations, and the kernel's error
  on that exact function;
* propagation: the same reads with every node and check propagated at twice the
  sampling, and where the worst pixels sit.

Usage::

    python docs/scripts/diagnose_floor.py DELIVERY CACHE

``CACHE`` holds the arrays written by ``propagate.py`` (and its ``level1`` mode).
Writes ``floor_diagnosis.json`` and ``floor_diagnosis.png`` under
``_static/validation``.
"""

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import eyepiece as ep  # noqa: E402
import hwostyle  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import yippy.sinc as sc  # noqa: E402
from yippy.sinc import kernel as sk  # noqa: E402
from yippy.sinc.reader import Interpolator, read_batch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import propagate as pr  # noqa: E402
import validation_figures as vf  # noqa: E402

H = 0.38
PITCH = 0.25
CENTER = 63.5
STAR_IWA = pr.IWA
# Declared entrance support (x, y) in D from the manifest; lambda0 = sampling_min, so
# the band edge in cycles per lambda0/D equals the support.
SUPPORT = {"x": 0.946, "y": 0.982}


def brightest_map(nodes):
    """Each pixel's brightest stored value over the quadrant and its three flips."""
    stored = np.asarray(nodes).astype(float).max(axis=0)
    return np.maximum.reduce(
        [stored, stored[:, ::-1], stored[::-1, :], stored[::-1, ::-1]]
    )


def make_reader(grid, arguments):
    """An Interpolator over ``grid`` with the delivery's domain and memory budget."""
    return Interpolator(
        grid,
        arguments["domain"],
        sc.KernelProfile(),
        sc.ReadPolicy(),
        sc.Tiling(pixel_tile=2048),
        arguments["memory"],
    )


def with_nodes(grid, nodes, indices):
    """The admitted grid with float64 ``nodes`` [N, 128, 128] as its samples."""
    ix, iy = np.asarray(indices).T
    shape = grid.intensity.shape
    lattice = np.zeros(shape, dtype=np.float64)
    lattice[0, iy, ix] = nodes
    return eqx.tree_at(lambda g: g.intensity, grid, jnp.asarray(lattice))


def read_all(reader, xy):
    """Read every position [Q, 2] and return the float64 images [Q, ny, nx]."""
    batch = eqx.filter_jit(read_batch)
    product = jnp.zeros(len(xy), dtype=jnp.int32)
    return np.asarray(batch(reader, jnp.asarray(xy), product).intensity)


def kernel_tone_error(freqs):
    """Largest error of one read of a unit complex tone at each frequency.

    The tone is sampled at the lattice nodes and read back with the 32-tap
    kernel at 64 sub-node offsets; returns the maximum over offsets and over
    the node phase of the absolute error, per frequency (cycles per lambda0/D).
    """
    offsets = (np.arange(64) + 0.5) / 64 * H
    out = []
    for f in freqs:
        worst = 0.0
        for x in offsets:
            floor = int(np.floor(x / H))
            taps = floor + np.arange(-15, 17)
            w = np.asarray(sk.weights(jnp.asarray(x / H - taps)))
            read = np.sum(w * np.exp(2j * np.pi * f * taps * H))
            worst = max(worst, abs(read - np.exp(2j * np.pi * f * x)))
        out.append(worst)
    return np.array(out)


def kaiser_spectrum(values, step, beta=20.0):
    """Amplitude spectrum of ``values`` under a Kaiser window, and the lobe width."""
    window = np.kaiser(len(values), beta)
    spectrum = np.fft.rfft(values * window)
    freqs = np.fft.rfftfreq(len(values), step)
    half_width = np.sqrt(1 + (beta / np.pi) ** 2) / (len(values) * step)
    return freqs, np.abs(spectrum) / window.sum(), half_width


def edge(f, amp, threshold):
    """Highest frequency whose amplitude exceeds ``threshold`` of the peak."""
    return float(f[amp > threshold * amp.max()].max())


def region_labels(shape, source):
    """Region of every pixel: behind the inner working angle, dark hole, corner."""
    yy, xx = np.indices(shape)
    x, y = (xx - CENTER) * PITCH, (yy - CENTER) * PITCH
    r_star = np.hypot(x, y)
    r_source = np.hypot(x - source[0], y - source[1])
    return x, y, r_star, r_source


def main(root, cache, out_dir):
    """Run the three tests, write the JSON and the figure."""
    cache = Path(cache)
    arguments, check_xy, truth, _, _ = vf.load(root)
    delivered = np.asarray(arguments["intensity_nodes"][0])
    brightest = brightest_map(delivered)
    admission = sc.admit_grid(**arguments)
    grid32 = admission.require()
    nodes64 = np.load(cache / "nodes64.npy")
    result = {}

    # --- (a) storage precision -------------------------------------------------
    grid64 = with_nodes(grid32, nodes64, arguments["indices_xy"])
    reads32 = read_all(make_reader(grid32, arguments), check_xy)
    reads64 = read_all(make_reader(grid64, arguments), check_xy)
    pix32 = (np.abs(reads32 - truth) / brightest).max(axis=(1, 2))
    pix64 = (np.abs(reads64 - truth) / brightest).max(axis=(1, 2))
    quant = np.abs(delivered.astype(float) - nodes64) / brightest
    result["storage"] = dict(
        delivered_dtype=str(delivered.dtype),
        repropagated_float64_casts_to_delivered=bool(
            np.array_equal(nodes64.astype(np.float32), delivered)
        ),
        node_quantization_over_brightest_max=float(quant.max()),
        float32_unit_roundoff=float(2.0**-24),
        worst_pixel_relative_float32_nodes=float(pix32.max()),
        worst_pixel_relative_float64_nodes=float(pix64.max()),
        read_change_float32_to_float64_over_brightest_max=float(
            (np.abs(reads32 - reads64) / brightest).max()
        ),
    )

    # --- where the worst pixels sit ------------------------------------------
    worst_check = int(np.argmax(pix32))
    source = check_xy[worst_check]
    rel = np.abs(reads32 - truth) / brightest
    x, y, r_star, r_source = region_labels(rel.shape[1:], source)
    top = np.argsort(rel[worst_check].ravel())[::-1][:12]
    peak_local = vf._peak(truth[worst_check], source, (CENTER, CENTER), PITCH)
    worst_pixels = []
    for flat in top:
        iy, ix = divmod(int(flat), rel.shape[2])
        worst_pixels.append(
            dict(
                ix=ix,
                iy=iy,
                x_lod=float(x[iy, ix]),
                y_lod=float(y[iy, ix]),
                r_from_star_lod=float(r_star[iy, ix]),
                r_from_source_lod=float(r_source[iy, ix]),
                brightest_over_local_peak=float(brightest[iy, ix] / peak_local),
                error_over_brightest=float(rel[worst_check, iy, ix]),
            )
        )
    worst_check_r = {}
    regions = {
        "behind the inner working angle (r < 6.72)": r_star < STAR_IWA,
        "dark hole (6.72 <= r < 16)": (r_star >= STAR_IWA) & (r_star < 16),
        "detector corners (r >= 16)": r_star >= 16,
    }
    for name, mask in regions.items():
        worst_check_r[name] = float(rel[:, mask].max())
    dim_bins = {}
    decades = np.floor(np.log10(brightest / brightest.max()))
    for d in np.unique(decades):
        sel = decades == d
        dim_bins[f"{int(d)}"] = dict(
            pixels=int(sel.sum()),
            worst_error_over_brightest=float(rel[:, sel].max()),
        )
    result["location"] = dict(
        worst_check_index=worst_check,
        worst_check_xy_lod=[float(source[0]), float(source[1])],
        worst_pixels=worst_pixels,
        worst_error_by_region_all_checks=worst_check_r,
        worst_error_by_brightness_decade=dim_bins,
    )

    # --- (b) band and the kernel's error on the exact function ---------------
    px = pr.LINE_STEP
    bands = {}
    top_pixels = [(p["iy"], p["ix"]) for p in worst_pixels[:8]]
    tone_f = np.linspace(0.02, 1.2, 60)
    tone_error = kernel_tone_error(tone_f)
    for axis, support in SUPPORT.items():
        line = np.load(cache / f"line_{axis}.npy")
        rows = []
        for iy, ix in top_pixels:
            g = line[:, iy, ix]
            f, amp, lobe = kaiser_spectrum(g, px)
            guard = support + lobe
            out = np.sqrt((amp[f > guard] ** 2).sum() / (amp**2).sum())
            # Wiener norm: the sum of tone amplitudes, which sets how a per-tone
            # kernel error scales onto the pixel's response.
            raw = np.fft.rfft(g) / len(g)
            fr = np.fft.rfftfreq(len(g), px)
            tone = np.interp(fr, tone_f, tone_error)
            wiener = 2 * np.abs(raw).sum()
            predicted = 2 * (np.abs(raw) * tone)[fr <= support].sum()
            node_s = g[:: round(H / px)]
            sup_nodes = np.abs(node_s).max()
            rows.append(
                dict(
                    ix=ix,
                    iy=iy,
                    out_of_band_rms_fraction=float(out),
                    highest_frequency_above_1e6_of_peak=edge(f, amp, 1e-6),
                    highest_frequency_above_1e8_of_peak=edge(f, amp, 1e-8),
                    wiener_norm_over_sup=float(wiener / np.abs(g).max()),
                    predicted_read_error_over_sup=float(predicted / np.abs(g).max()),
                    sup_over_brightest=float(np.abs(g).max() / brightest[iy, ix]),
                    sup_nodes_over_sup=float(sup_nodes / np.abs(g).max()),
                )
            )
        bands[axis] = dict(
            declared_support_D=support, guard_lobe_halfwidth=float(lobe), pixels=rows
        )
    result["band"] = dict(
        declared_half_band=SUPPORT,
        window="Kaiser beta 20",
        lines=bands,
        kernel_tone_error_max_in_band=float(
            tone_error[tone_f <= max(SUPPORT.values())].max()
        ),
    )

    # --- the kernel on the exact function, query by query ----------------------
    ratio = round(H / px)
    sweeps = {}
    offsets_xy = np.asarray(arguments["offsets_xy_lod"])
    lattice_extent = {
        "x": float(offsets_xy[:, 0].max()),
        "y": float(offsets_xy[:, 1].max()),
    }
    for axis, limit in (("x", 4.18), ("y", 5.70)):
        line = np.load(cache / f"line_{axis}.npy")
        center = line.shape[0] // 2
        half = center // ratio
        node_vals = line[center + ratio * np.arange(-half, half + 1)]
        sup = np.abs(line).max(axis=0)
        positions, over_brightest, over_sup = [], [], []
        worst_pixel = None
        for query in range(center, center + int(limit / px)):
            if (query - center) % ratio == 0:
                continue
            s = (query - center) * px
            taps = int(np.floor(s / H)) + np.arange(-15, 17)
            w = np.asarray(sk.weights(jnp.asarray(s / H - taps)))
            err = np.abs(
                np.tensordot(w, node_vals[taps + half], axes=(0, 0)) - line[query]
            )
            rb, rs = err / brightest, err / sup
            positions.append(float(s))
            over_brightest.append(float(rb.max()))
            over_sup.append(float(rs.max()))
            if worst_pixel is None or rb.max() > worst_pixel[0]:
                worst_pixel = (
                    float(rb.max()),
                    np.unravel_index(rb.argmax(), rb.shape),
                    float(rs[np.unravel_index(rb.argmax(), rb.shape)]),
                    float(s),
                )
        iy_w, ix_w = worst_pixel[1]
        peak_pos = float((np.argmax(np.abs(line[:, iy_w, ix_w])) - center) * px)
        sweeps[axis] = dict(
            source_position_lod=positions,
            max_error_over_brightest=over_brightest,
            max_error_over_pixel_peak=over_sup,
            served_cell_start_lod=3.8 if axis == "x" else 5.32,
            worst_error_over_brightest_in_cell=float(
                max(
                    v
                    for v, p in zip(over_brightest, positions, strict=False)
                    if p >= (3.8 if axis == "x" else 5.32)
                )
            ),
            worst_error_over_pixel_peak_in_cell=float(
                max(
                    v
                    for v, p in zip(over_sup, positions, strict=False)
                    if p >= (3.8 if axis == "x" else 5.32)
                )
            ),
            worst_pixel_ix_iy=[int(worst_pixel[1][1]), int(worst_pixel[1][0])],
            pixel_peak_over_brightest_at_worst_pixel=float(
                sup[worst_pixel[1]] / brightest[worst_pixel[1]]
            ),
            worst_pixel_error_over_brightest=worst_pixel[0],
            worst_pixel_error_over_pixel_peak=worst_pixel[2],
            worst_pixel_query_position_lod=worst_pixel[3],
            worst_pixel_peak_position_lod=peak_pos,
            scanned_line_extent_lod=float(center * px),
            stored_lattice_extent_lod=float(lattice_extent[axis]),
            peak_inside_stored_lattice=bool(abs(peak_pos) <= lattice_extent[axis]),
            worst_error_over_pixel_peak_all_pixels_all_queries=float(max(over_sup)),
        )
    result["kernel_on_exact_function"] = dict(
        note="queries at h/4 and h/2 and 3h/4 offsets along each axis, all pixels",
        sweeps=sweeps,
    )

    # --- (c) propagation resolution -----------------------------------------
    p = cache / "nodes_level1.npy"
    if p.exists():
        nodes1 = np.load(p)
        truth1 = np.load(cache / "checks_level1.npy")
        grid1 = with_nodes(grid32, nodes1, arguments["indices_xy"])
        ids = pr.LEVEL1_CHECKS
        reads1 = read_all(make_reader(grid1, arguments), check_xy[ids])
        bright1 = brightest_map(nodes1)
        e1 = (np.abs(reads1 - truth1) / bright1).max(axis=(1, 2))
        e0 = pix64[ids]
        diff_model = (np.abs(nodes1 - nodes64) / bright1).max()
        result["propagation_level1"] = dict(
            checks=[f"check-{i:03}" for i in ids],
            error_over_brightest_level0_float64_nodes=[float(v) for v in e0],
            error_over_brightest_level1_nodes=[float(v) for v in e1],
            node_level0_vs_level1_over_brightest_max=float(diff_model),
        )
    ref = cache / "refine_level2.npy"
    if ref.exists():
        l0, l1, l2 = (np.load(cache / f"refine_level{k}.npy") for k in range(3))
        pts = np.load(cache / "refine_positions.npy")
        pk = np.array(
            [vf._peak(l0[i], p_, (CENTER, CENTER), PITCH) for i, p_ in enumerate(pts)]
        )
        names = ["worst check"] + [f"node {ix},{iy}" for ix, iy in pr.REFINE_NODES]
        rows = []
        for i, name in enumerate(names):
            rows.append(
                dict(
                    position=name,
                    xy_lod=[float(pts[i, 0]), float(pts[i, 1])],
                    level0_vs_level1_over_local_peak=float(
                        np.abs(l0[i] - l1[i]).max() / pk[i]
                    ),
                    level1_vs_level2_over_local_peak=float(
                        np.abs(l1[i] - l2[i]).max() / pk[i]
                    ),
                    level0_vs_level2_over_local_peak=float(
                        np.abs(l0[i] - l2[i]).max() / pk[i]
                    ),
                )
            )
        result["propagation_levels"] = rows
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "floor_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    figure(result, out_dir)


def figure(result, out_dir):
    """Error against query position: brightest-stored scale against pixel-peak scale."""
    hwostyle.use("dark")
    palette = hwostyle.palette
    floor = result["band"]["kernel_tone_error_max_in_band"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    sweeps = result["kernel_on_exact_function"]["sweeps"]
    for ax, axis in zip(axes, ("x", "y"), strict=True):
        sweep = sweeps[axis]
        s = np.array(sweep["source_position_lod"])
        ax.axvspan(
            sweep["served_cell_start_lod"], s.max() + 0.05, color=ep.style.neutral(0.2)
        )
        ax.axhline(floor, color=ep.style.neutral(0.45), lw=1, ls="--")
        ax.plot(
            s,
            sweep["max_error_over_brightest"],
            ".-",
            color=palette.cyan,
            label="over the pixel's brightest stored value",
        )
        ax.plot(
            s,
            sweep["max_error_over_pixel_peak"],
            ".-",
            color=palette.pink,
            label="over the pixel's peak response",
        )
        ax.set_yscale("log")
        ax.set_xlabel(rf"source position along ${axis}$ ($\lambda_0/D$)")
        ax.set_title(f"reads along ${axis}$, worst pixel at each position")
    axes[0].set_ylabel("one-axis read error")
    axes[0].annotate(
        "kernel floor, one tone",
        (0.02, floor * 1.25),
        xycoords=("axes fraction", "data"),
        color=ep.style.neutral(0.5),
        fontsize=9,
    )
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "floor_diagnosis.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    main(Path(sys.argv[1]), sys.argv[2], here.parent / "_static" / "validation")
