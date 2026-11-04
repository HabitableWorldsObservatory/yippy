"""Measure circle-reader interpolation error without external data.

Run with PYTHONPATH=src python tools/check_circle_reader.py. CSV goes to stdout;
coordinate diagnostics go to stderr. All calculations use float64.
"""

import csv
import platform
import sys
from functools import partial
from time import perf_counter

import jax
import numpy as np
from hwoutils.map_coordinates import map_coordinates
from scipy.interpolate import RectBivariateSpline
from scipy.special import i0, j1

from yippy.performance import azimuthal_mean_profile

RADII = np.array(
    [
        2.5,
        3,
        3.03125,
        3.23,
        3.25,
        4,
        5,
        5.03125,
        6,
        7,
        8,
        9,
        10,
        10.03125,
        11,
    ]
)


def circle_coordinates(size, pixel_scale):
    """Independent full-circle coordinates at 2048 midpoint angles."""
    center = (size - 1) / 2
    theta = (np.arange(2048) + 0.5) * 2 * np.pi / 2048
    return (
        center + RADII[:, None] / pixel_scale * np.sin(theta),
        center + RADII[:, None] / pixel_scale * np.cos(theta),
    )


def airy(radius):
    """Unit-peak circular-pupil intensity, with radius in lambda/D."""
    z = np.pi * np.maximum(radius, 1e-20)
    return (2 * j1(z) / z) ** 2


def cases(pixel_scale):
    """Sample a radial Airy profile and a displaced Gaussian on pixel centers."""
    # Keep the physical field of view fixed when refining the detector grid.
    size = round(32 / pixel_scale)
    center = (size - 1) / 2
    y, x = (np.indices((size, size)) - center) * pixel_scale
    yield "airy", airy(np.hypot(y, x)), airy
    yield (
        "gaussian",
        np.exp(-((x - 1) ** 2 + (y + 2) ** 2) / 18),
        lambda r: np.exp(-(r * r + 5) / 18) * i0(r * np.sqrt(5) / 9),
    )


def measurements():
    """Compare the production reader, direct Keys sampling and a quintic spline."""
    radii = RADII
    for pixel_scale in (0.25, 0.125):
        for name, density, analytic_mean in cases(pixel_scale):
            size = density.shape[0]
            center = (size - 1) / 2
            # The reader returns per-pixel values. Convert to density once.
            image = density * pixel_scale**2
            reference = analytic_mean(radii)
            default = (
                np.asarray(
                    azimuthal_mean_profile(image, (center, center), radii / pixel_scale)
                )
                / pixel_scale**2
            )
            dense = (
                np.asarray(
                    azimuthal_mean_profile(
                        image, (center, center), radii / pixel_scale, n_theta=2048
                    )
                )
                / pixel_scale**2
            )
            # Independent angular coordinates: no arc construction/profile grid.
            coords = circle_coordinates(size, pixel_scale)
            direct = (
                np.asarray(
                    map_coordinates(image, coords, order=3, mode="nearest")
                ).mean(axis=1)
                / pixel_scale**2
            )
            spline = RectBivariateSpline(
                np.arange(size), np.arange(size), image, kx=5, ky=5
            )
            quintic = spline.ev(*coords).mean(axis=1) / pixel_scale**2
            coordinate_error = np.max(np.abs(dense / direct - 1))
            print(
                f"{name} pixel_scale={pixel_scale}: "
                f"max |reader/direct_keys - 1|={coordinate_error:.3g}",
                file=sys.stderr,
            )
            for i, radius in enumerate(radii):
                yield {
                    "map": name,
                    "pixel_scale_lod": pixel_scale,
                    "radius_lod": radius,
                    "analytic_mean": reference[i],
                    "default_error_pct": 100 * (default[i] / reference[i] - 1),
                    "dense_error_pct": 100 * (dense[i] / reference[i] - 1),
                    "quintic_error_pct": 100 * (quintic[i] / reference[i] - 1),
                }


def median_ms(operation):
    """Time seven warmed, synchronized calls; exclude compilation and warmup."""

    def call():
        value = operation()
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()

    call()
    elapsed = []
    for _ in range(7):
        start = perf_counter()
        call()
        elapsed.append(1000 * (perf_counter() - start))
    return float(np.median(elapsed))


def footprint_mean(evaluate, rows, cols, order=8):
    """Average over an axis-aligned one-pixel square and then over each circle."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    result = np.zeros_like(rows)
    for dy, wy in zip(nodes / 2, weights / 2, strict=True):
        for dx, wx in zip(nodes / 2, weights / 2, strict=True):
            result += wy * wx * evaluate(rows + dy, cols + dx)
    return result.mean(axis=1)


def alternatives():
    """Measure accuracy and CPU costs of alternatives on the same Airy scan.

    Timings go to stderr, separately from the CSV radius scan. These are local
    prototypes, not a proposed change to the production reader or its caches.
    """
    print(
        f"timings: {platform.machine()}, Python {platform.python_version()}, "
        f"JAX {jax.__version__}, backend={jax.default_backend()}; "
        "15 radii, float64, median of 7 warm calls",
        file=sys.stderr,
    )
    for pixel_scale in (0.25, 0.125):
        _, density, _ = next(cases(pixel_scale))
        size = density.shape[0]
        center = (size - 1) / 2
        coords = circle_coordinates(size, pixel_scale)
        grid = np.arange(size)
        build = partial(RectBivariateSpline, grid, grid, density, kx=5, ky=5)
        spline = build()
        image = jax.device_put(density)
        radii_pix = jax.device_put(RADII / pixel_scale)
        for n_theta in (2 * size, 2048):

            def read(image, radii_pix, center=center, n_theta=n_theta):
                return azimuthal_mean_profile(
                    image, (center, center), radii_pix, n_theta=n_theta
                )

            compiled = jax.jit(read)
            start = perf_counter()
            compiled(image, radii_pix).block_until_ready()
            first_ms = 1000 * (perf_counter() - start)
            eager_ms = median_ms(partial(read, image, radii_pix))
            jit_ms = median_ms(partial(compiled, image, radii_pix))
            print(
                f"cost scale={pixel_scale} Keys angles={n_theta}: "
                f"eager_ms={eager_ms:.4f}, jit_ms={jit_ms:.4f}, "
                f"jit_first_ms={first_ms:.2f}",
                file=sys.stderr,
            )

        def spline_read(spline=spline, coords=coords):
            return spline.ev(*coords).mean(axis=1)

        def native_map(pixel_scale=pixel_scale):
            return next(cases(pixel_scale))

        print(
            f"cost scale={pixel_scale} quintic: build_ms={median_ms(build):.4f}, "
            f"read_ms={median_ms(spline_read):.4f}; "
            f"native_airy_map_ms={median_ms(native_map):.4f}",
            file=sys.stderr,
        )
        if pixel_scale != 0.25:
            continue

        # Upsample the existing coarse spline, rather than regenerate the model.
        fine_grid = (np.arange(2 * size) - (2 * size - 1) / 2) / 2 + center
        upsample = partial(spline, fine_grid, fine_grid)
        fine = upsample()
        fine_center = (2 * size - 1) / 2
        upsampled = np.asarray(
            azimuthal_mean_profile(
                fine, (fine_center, fine_center), RADII / 0.125, n_theta=2048
            )
        )
        print(
            f"cost coarse-quintic-to-fine-grid: prep_ms={median_ms(upsample):.4f}; "
            "then use the fine-grid Keys read cost",
            file=sys.stderr,
        )

        def exact(rows, cols, center=center, pixel_scale=pixel_scale):
            return airy(np.hypot(rows - center, cols - center) * pixel_scale)

        footprint = footprint_mean(spline.ev, *coords)
        footprint_ref = footprint_mean(exact, *coords)
        footprint_ref12 = footprint_mean(exact, *coords, order=12)
        footprint_ms = median_ms(partial(footprint_mean, spline.ev, *coords))
        print(
            "footprint quadrature max |8x8/12x12 - 1|="
            f"{np.max(np.abs(footprint_ref / footprint_ref12 - 1)):.3g}; "
            f"read_ms={footprint_ms:.4f}",
            file=sys.stderr,
        )
        for i, radius in enumerate(RADII):
            print(
                f"alternative radius={radius:.8g}: "
                f"upsampled_error_pct={100 * (upsampled[i] / airy(radius) - 1):.8g}, "
                "footprint_point_bias_pct="
                f"{100 * (footprint_ref[i] / airy(radius) - 1):.8g}, "
                "footprint_quintic_error_pct="
                f"{100 * (footprint[i] / footprint_ref[i] - 1):.8g}",
                file=sys.stderr,
            )


def main():
    """Print a reproducible radius scan; neither load nor save cached products."""
    with jax.enable_x64():
        rows = list(measurements())
        alternatives()
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
