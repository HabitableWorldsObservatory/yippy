"""Regenerate aperture accuracy evidence using analytic and current-code references.

Run from a source checkout with its src directory on PYTHONPATH::

    python tools/benchmark_apertures.py output/aperture-convergence

Use --airy-only for an offline analytic check. No historical Git objects or
imports from the test suite are required. See docs/benchmarking.md for provenance.
"""

import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from optixstuff import performance as perf
from scipy.optimize import brentq
from scipy.special import j0, j1

from yippy import __version__, fetch_yip
from yippy._yip import _YIPData
from yippy.datasets import CATALOG, DATA_RELEASE_TAG
from yippy.performance import _iter_xaxis_positions


def write_table(directory, name, header, rows):
    """Persist numeric evidence as a CSV with units in the column names."""
    with (directory / name).open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def airy_psf(pixel_lod, shift_pix=0.0, size=256):
    """Unit-flux Airy samples without renormalizing the finite crop."""
    rows, cols = np.indices((size, size))
    x = np.pi * np.hypot(rows - size // 2, cols - size // 2 - shift_pix) * pixel_lod
    amplitude = 2 * j1(x) / np.where(x == 0, 1.0, x)
    amplitude[x == 0] = 1.0
    return np.pi / 4 * amplitude**2 * pixel_lod**2


def encircled_energy(radius_lod):
    """Exact infinite-image Airy encircled energy (Born and Wolf 8.5.2)."""
    x = np.pi * radius_lod
    return 1 - j0(x) ** 2 - j1(x) ** 2


def measure(psf, pixel_lod, factor, ratio, center):
    """Return circle flux/area and truncation flux/area at one resolution."""
    with jax.enable_x64():
        image = jnp.asarray(psf, dtype=jnp.float64)
        cf, ca, _ = perf.circular_aperture(
            image,
            0.7 / pixel_lod,
            pixel_lod,
            oversample=factor,
            half_width_pix=int(np.ceil(1.7 / pixel_lod)) + 1,
            nominal_center=center,
        )
        tf, ta = perf.truncation_core(
            image,
            ratio,
            pixel_lod,
            oversample=factor,
            half_width_pix=perf.core_window_half_width_pix(pixel_lod),
            nominal_center=center,
        )
        return [float(v) for v in (cf, ca, tf, ta)]


def main(directory, airy_only=False):
    """Measure all requested gates, including every native shipped offset."""
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {
        "yippy_version": __version__,
        "dependencies": {
            name: version(name) for name in ("jax", "numpy", "scipy", "hwoutils")
        },
        "oversample": [10, 40],
        "references": {
            "airy": "Born and Wolf, Principles of Optics, 7th ed., section 8.5.2",
            "shipped": {
                "name": "eac1_aavc_2d",
                "release": DATA_RELEASE_TAG,
                "archive_md5": CATALOG["eac1_aavc_2d"]["md5"],
            },
        },
        "settings": {
            "circle_radius_lod": 0.7,
            "shipped_truncation_ratio": 0.3,
            "peak_search_radius_lod": perf.PEAK_SEARCH_RADIUS_LOD,
            "core_window_lod": perf.CORE_WINDOW_LOD,
            "airy_size": 256,
            "airy_pixel_lod": [0.25, 0.5],
            "airy_shift_pix": [0.0, 0.3],
            "precision": "float64",
        },
        "airy_only": airy_only,
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    half_max = brentq(lambda r: (2 * j1(np.pi * r) / (np.pi * r)) ** 2 - 0.5, 0.1, 1)
    airy_rows = []
    for pixel_lod in (0.25, 0.5):
        for shift in (0.0, 0.3):
            psf = airy_psf(pixel_lod, shift)
            for radius in (0.7, 1.0, 1.22):
                with jax.enable_x64():
                    flux = float(
                        perf.circular_aperture(
                            jnp.asarray(psf),
                            radius / pixel_lod,
                            pixel_lod,
                            nominal_center=(128.0, 128.0),
                        )[0]
                    )
                airy_rows.append(
                    [
                        pixel_lod,
                        shift,
                        "circle_flux",
                        radius,
                        flux,
                        encircled_energy(radius),
                    ]
                )
            values = measure(psf, pixel_lod, 10, 0.5, (128, 128))
            for metric, measured, analytic in (
                ("half_max_flux", values[2], encircled_energy(half_max)),
                ("half_max_area", values[3], np.pi * half_max**2),
            ):
                airy_rows.append(
                    [pixel_lod, shift, metric, half_max, measured, analytic]
                )
    write_table(
        directory,
        "airy.csv",
        ["pixel_lod", "shift_px", "quantity", "radius_lod", "measured_10x", "analytic"],
        airy_rows,
    )

    if airy_only:
        print(f"Wrote analytic Airy evidence to {directory}.")
        return

    coro = _YIPData(fetch_yip("eac1_aavc_2d"))
    positions = list(_iter_xaxis_positions(coro))
    pixel_lod = float(coro.grid.step)
    ny, nx = coro.grid.shape
    inside = [pos for pos in positions if 0 <= pos.py < ny and 0 <= pos.px < nx]
    shipped = min(inside, key=lambda pos: abs(pos.separation - 20))
    convergence = []
    for name, psf, scale, ratio, center in (
        ("Airy", airy_psf(0.5, 0.3), 0.5, 0.5, (128, 128)),
        ("eac1_aavc_2d", shipped.psf, pixel_lod, 0.3, (shipped.py, shipped.px)),
    ):
        for factor in (5, 10, 20, 40):
            convergence.append(
                [name, factor, *measure(psf, scale, factor, ratio, center)]
            )
    quantities = ["circle_flux", "circle_area_lod2", "core_flux", "core_area_lod2"]
    write_table(
        directory, "convergence.csv", ["case", "oversample", *quantities], convergence
    )

    all_offsets = []
    for pos in positions:
        values = [
            measure(pos.psf, pixel_lod, factor, 0.3, (pos.py, pos.px))
            for factor in (10, 40)
        ]
        all_offsets.append([pos.separation, *values[0], *values[1]])
    write_table(
        directory,
        "shipped_gates.csv",
        [
            "offset_lod",
            *[q + "_10x" for q in quantities],
            *[q + "_40x" for q in quantities],
        ],
        all_offsets,
    )
    print(
        f"Wrote tables to {directory}; shipped convergence offset {shipped.separation}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--airy-only",
        action="store_true",
        help="Run analytic cases without downloading a YIP.",
    )
    args = parser.parse_args()
    main(args.directory, airy_only=args.airy_only)
