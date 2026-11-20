"""Qualify the delivery beyond the 16 starter checks, by radius bin.

Reads the delivery with yippy.sinc at three groups of positions inside the served cell,
compares each read with a direct float64 propagation made with the delivery's own
propagator settings, and repeats the comparison for yippy's current blend
interpolation (``OffJAX``, inverse-distance weights over the four nearest delivered
images, each shifted to the query by a Fourier phase ramp).

Groups (all inside the served cell, none on a node):

* the 16 starter checks (stored truth);
* 40 positions behind the inner working angle (r < 6.72 lambda0/D);
* 40 positions from the inner working angle to 0.2 lambda0/D above it;
* 200 positions uniform over the cell, seeded.

Three metrics per read, each from the full 128 by 128 image:

* ``local_peak``: largest absolute pixel difference over the brightest direct
  pixel within 1 lambda0/D of the source;
* ``pixel_brightest``: the largest, over pixels, of the absolute difference over
  that pixel's brightest stored value (quadrant and flips);
* ``photometric``: relative error of the core flux, the sum of the pixels whose
  centers lie within 0.7 lambda0/D of the true source position.

Usage::

    python docs/scripts/qualify_extended.py DELIVERY CACHE

Writes ``qualification_extended.json``, ``qualification_extended.csv`` and
``qualification_radius.png`` under ``_static/validation``.
"""

import csv
import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import eyepiece as ep  # noqa: E402
import hwostyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import yippy.sinc as sc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_floor as df  # noqa: E402
import propagate as pr  # noqa: E402
import validation_figures as vf  # noqa: E402

PITCH, CENTER, APERTURE = 0.25, 63.5, 0.7
BINS = (
    ("behind the inner working angle", 0.0, pr.IWA),
    ("inner working angle to +0.2", pr.IWA, pr.NEAR_EDGE),
    ("beyond the inner working angle", pr.NEAR_EDGE, np.inf),
)
GROUPS = ("starter", "behind", "near", "random")


def aperture_flux(image, source):
    """Sum of the pixels whose centers lie within the aperture around ``source``."""
    yy, xx = np.indices(image.shape)
    x, y = (xx - CENTER) * PITCH, (yy - CENTER) * PITCH
    return image[(x - source[0]) ** 2 + (y - source[1]) ** 2 <= APERTURE**2].sum()


def metrics(reads, truth, xy, brightest):
    """The three metrics for each read, arrays [Q]."""
    err = np.abs(reads - truth)
    peak = np.array(
        [vf._peak(truth[i], xy[i], (CENTER, CENTER), PITCH) for i in range(len(xy))]
    )
    flux_t = np.array([aperture_flux(truth[i], xy[i]) for i in range(len(xy))])
    flux_r = np.array([aperture_flux(reads[i], xy[i]) for i in range(len(xy))])
    return dict(
        local_peak=err.max(axis=(1, 2)) / peak,
        pixel_brightest=(err / brightest).max(axis=(1, 2)),
        photometric=np.abs(flux_r - flux_t) / flux_t,
    )


def blend_reads(root, xy, chunk=40):
    """Blend reads (yippy interpolation) of every position [Q, 2], in chunks."""
    from lod_unit import lod

    from yippy.offjax import OffJAX

    off = OffJAX(
        root / "mono-000",
        "offax_psf.fits",
        "offax_psf_offset_list.fits",
        PITCH * lod,
        False,
        False,
    )
    out = []
    for start in range(0, len(xy), chunk):
        part = xy[start : start + chunk]
        out.append(np.asarray(off.create_psfs(part[:, 0], part[:, 1])))
    return np.concatenate(out)


def summarize(values):
    """Count, median, 95th percentile and worst of ``values``."""
    values = np.asarray(values)
    return dict(
        n=int(values.size),
        median=float(np.median(values)),
        p95=float(np.percentile(values, 95)),
        worst=float(values.max()),
    )


def main(root, cache, out_dir):
    """Read, score and tabulate every position; write the JSON, CSV and figure."""
    cache = Path(cache)
    arguments, check_xy, check_truth, _, _ = vf.load(root)
    delivered = np.asarray(arguments["intensity_nodes"][0])
    brightest = df.brightest_map(delivered)
    grid = sc.admit_grid(**arguments).require()
    reader = df.make_reader(grid, arguments)

    xy = {"starter": check_xy}
    truth = {"starter": check_truth}
    for name in GROUPS[1:]:
        xy[name] = np.load(cache / f"positions_{name}.npy")
        truth[name] = np.load(cache / f"truth_{name}.npy")
    all_xy = np.concatenate([xy[g] for g in GROUPS])
    all_truth = np.concatenate([truth[g] for g in GROUPS])
    group = np.concatenate([[g] * len(xy[g]) for g in GROUPS])
    sinc = df.read_all(reader, all_xy)
    blend = blend_reads(root, all_xy)
    # The blend routine must reproduce the delivery's own 16 blend reads.
    stored = np.load(root / "reads.npz")["yippy"]
    consistent = float(np.abs(blend[: len(check_xy)] - stored).max())
    m_sinc = metrics(sinc, all_truth, all_xy, brightest)
    m_blend = metrics(blend, all_truth, all_xy, brightest)
    radius = np.hypot(all_xy[:, 0], all_xy[:, 1])

    import yippy

    result = dict(
        yippy_version=getattr(yippy, "__version__", "unknown"),
        seed=pr.SEED,
        groups={g: len(xy[g]) for g in GROUPS},
        aperture_radius_lod=APERTURE,
        design_iwa_lod=pr.IWA,
        blend_matches_delivery_reads_max_abs=consistent,
        bins=[],
    )
    for name, lo, hi in BINS:
        sel = (radius >= lo) & (radius < hi)
        entry = dict(
            name=name,
            r_min=lo,
            r_max=None if np.isinf(hi) else hi,
            n=int(sel.sum()),
            sinc={k: summarize(v[sel]) for k, v in m_sinc.items()},
            blend={k: summarize(v[sel]) for k, v in m_blend.items()},
        )
        result["bins"].append(entry)
    rnd = group == "random"
    result["random_200"] = dict(
        sinc={k: summarize(v[rnd]) for k, v in m_sinc.items()},
        blend={k: summarize(v[rnd]) for k, v in m_blend.items()},
    )
    result["all"] = dict(
        n=len(all_xy),
        sinc={k: summarize(v) for k, v in m_sinc.items()},
        blend={k: summarize(v) for k, v in m_blend.items()},
        worst_sinc_position=all_xy[int(np.argmax(m_sinc["pixel_brightest"]))].tolist(),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qualification_extended.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    with open(out_dir / "qualification_extended.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["group", "x_lod", "y_lod", "r_lod"]
            + [f"{m}_{k}" for m in ("sinc", "blend") for k in m_sinc]
        )
        for i in range(len(all_xy)):
            writer.writerow(
                [
                    group[i],
                    f"{all_xy[i, 0]:.6f}",
                    f"{all_xy[i, 1]:.6f}",
                    f"{radius[i]:.6f}",
                ]
                + [f"{mm[k][i]:.4e}" for mm in (m_sinc, m_blend) for k in m_sinc]
            )
    print(json.dumps(result, indent=2))

    # Figure: error against radius for each metric, yippy.sinc beside the blend.
    hwostyle.use("dark")
    palette = hwostyle.palette
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True)
    titles = {
        "local_peak": "over the local peak",
        "pixel_brightest": "over each pixel's brightest value",
        "photometric": "core flux, 0.7 radius aperture",
    }
    for ax, key in zip(axes, titles, strict=True):
        ax.scatter(radius, m_blend[key], s=6, color=palette.pink, alpha=0.6, lw=0)
        ax.scatter(radius, m_sinc[key], s=6, color=palette.cyan, alpha=0.8, lw=0)
        ax.axvline(pr.IWA, color=ep.style.neutral(0.4), lw=1)
        ax.set_yscale("log")
        ax.set_title(titles[key])
        ax.set_xlabel(r"source radius ($\lambda_0/D$)")
    axes[0].set_ylabel("relative error of the read")
    axes[0].text(
        6.76,
        axes[0].get_ylim()[0] * 3,
        "design IWA",
        color=ep.style.neutral(0.5),
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "qualification_radius.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    main(Path(sys.argv[1]), sys.argv[2], here.parent / "_static" / "validation")
