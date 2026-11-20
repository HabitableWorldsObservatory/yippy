"""Regenerate the validation page figures and numbers from a sinc delivery.

The input is a ``sinc-delivery-0.1`` directory, such as the physicaloptix
LUVOIR-A APLC starter delivery. The script admits the quadrant grid with its
mirror and interior checks, reads every interior check position, and writes
``docs/_static/validation/validation.json`` and two figures. An optional
``reads.npz`` in the delivery with a ``yippy`` array of blend reads at the
check positions adds the blend comparison.

Usage::

    python docs/scripts/validation_figures.py DELIVERY_DIR

Requires astropy, eyepiece and hwostyle in addition to yippy.sinc.
"""

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import jax
from optixstuff.grids import Frame, PixelGrid

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import hwostyle  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from hwoutils.constants import m2nm  # noqa: E402

import yippy.sinc as sc  # noqa: E402
from yippy.sinc import viz  # noqa: E402
from yippy.sinc.geometry import source_grid  # noqa: E402
from yippy.sinc.grid import (  # noqa: E402
    CheckSet,
    MirrorCheckSet,
    Normalization,
    Reflection,
)
from yippy.sinc.reader import Interpolator, read_batch  # noqa: E402

PREMISES = (
    "entrance_tilt_model",
    "fixed_linear_optics",
    "fixed_detector_registration",
    "fixed_pixel_response",
    "source_independent_normalization",
    "source_independent_spectral_weights",
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def _peak(image, source, center, pitch, radius=1.0):
    yy, xx = np.indices(image.shape)
    x = (xx - center[0]) * pitch
    y = (yy - center[1]) * pitch
    return image[(x - source[0]) ** 2 + (y - source[1]) ** 2 <= radius**2].max()


def load(root):
    """Translate a sinc delivery into ``admit_grid`` arguments."""
    manifest = json.loads((root / "sinc-delivery.json").read_text())
    plane = manifest["planes"][0]
    psf_path, offsets_path = root / plane["psf_path"], root / plane["offsets_path"]
    if _sha(psf_path) != plane["psf_sha256"]:
        raise ValueError("training image hash mismatch")
    if _sha(offsets_path) != plane["offsets_sha256"]:
        raise ValueError("offset hash mismatch")
    cube = fits.getdata(psf_path).astype(np.float32)
    offsets = fits.getdata(offsets_path).astype(np.float64)
    training = _rows(root / manifest["training_coordinates_path"])
    order = np.array([int(r["index"]) for r in training])
    indices = np.zeros((len(training), 2), dtype=np.int32)
    indices[order] = [[int(r["ix"]), int(r["iy"])] for r in training]
    ny, nx = manifest["detector_shape_yx"]
    pitch = manifest["pixel_scale_lod_xy"][0]
    center = manifest["detector_center_xy"]
    wavelength_nm = plane["wavelength_m"] * m2nm
    premises = manifest["premises"]
    products = sc.Products(
        pixels=PixelGrid(
            (ny, nx),
            (jnp.array(center, float))[0],
            (jnp.array(center, float))[1],
            jnp.eye(2) * pitch,
            "lod",
            Frame(
                manifest["lambda0_m"] * m2nm,
                manifest["diameter_m"],
                manifest["diameter_convention"],
                manifest["coordinate_model"],
            ),
        ),
        normalization=Normalization(
            "incident_photon",
            jnp.array([plane["photon_conversion"]]),
            jnp.array([np.nan]),
            (),
            "delivery-normalization",
        ),
        kind="monochromatic",
        wavelength_nm=jnp.array([wavelength_nm]),
        limits_nm=jnp.array([[wavelength_nm, wavelength_nm]]),
        sampling_min_nm=jnp.array([plane["sampling_min_wavelength_m"] * m2nm]),
        campaign_sampling_min_nm=jnp.array(np.nan),
        support_extent_xy=jnp.array([manifest["support_extent_D"]]),
        support_evidence=jnp.array([1], dtype=jnp.int32),
        support_source_ids=(manifest["support_evidence"],),
        premises=sc.Premises(
            jnp.array([premises[p]["value"] for p in PREMISES]),
            jnp.array([premises[p]["evidence"] for p in PREMISES], dtype=jnp.int32),
            tuple(premises[p]["source_id"] for p in PREMISES),
        ),
        optical_id=manifest["optical_state"]["state_id"],
        spectral_weight_id="monochromatic",
        identity=f"{manifest['delivery_id']}:{plane['id']}",
    )
    lattice = source_grid(
        jnp.array(manifest["source_origin_lod"], dtype=jnp.float64),
        jnp.array(manifest["source_spacing_lod"], dtype=jnp.float64),
        (int(indices[:, 1].max()) + 1, int(indices[:, 0].max()) + 1),
        products.frame,
    )
    reflection_meta = manifest["reflection"]
    argument = (
        reflection_meta["response_argument_kind"]
        + reflection_meta["response_argument"]
        + manifest["optical_state"]["state_id"]
    )
    reflection = Reflection(
        reflection_meta["kind"],
        jnp.array([True, True]),
        reflection_meta["declaration_id"],
        hashlib.sha256(argument.encode()).hexdigest(),
        jnp.int32(1),
        "computed-at-admission",
        jnp.array([False]),
    )
    bounds = manifest["served"]["bounds_lod"]
    domain = sc.Domain(
        manifest["served"]["kind"],
        manifest["served"]["boundary"],
        jnp.array([[bounds[0][0], bounds[1][0]], [bounds[0][1], bounds[1][1]]]),
        jnp.zeros(2),
    )
    mirror_rows = _rows(root / "starter_mirror_checks.csv")
    mirror_images = fits.getdata(root / manifest["mirror_checks"][0]["psf_path"])
    mirror = MirrorCheckSet(
        jnp.array([[float(r["x_lod"]), float(r["y_lod"])] for r in mirror_rows]),
        jnp.array(
            [[int(r["source_ix"]), int(r["source_iy"])] for r in mirror_rows],
            dtype=jnp.int32,
        ),
        jnp.asarray(
            mirror_images.astype(np.float32)[[int(r["index"]) for r in mirror_rows]]
        ),
        jnp.zeros(len(mirror_rows), dtype=jnp.int32),
        "delivery-mirror-checks",
    )
    check_rows = _rows(root / "starter_checks.csv")
    check_images = fits.getdata(root / "checks/mono-000-float64.fits").astype(float)
    check_xy = np.array([[float(r["x_lod"]), float(r["y_lod"])] for r in check_rows])
    checks = CheckSet(
        jnp.asarray(check_xy),
        jnp.asarray(check_images),
        jnp.zeros(len(check_rows), dtype=jnp.int32),
        jnp.asarray([c["local_peak_reference"] for c in manifest["checks"]]),
        "delivery-checks",
        "direct-propagation-float64",
        "finite-model",
    )
    gib = 2**30
    arguments = dict(
        intensity_nodes=jnp.asarray(cube[None]),
        offsets_xy_lod=jnp.asarray(offsets),
        declared_lattice=lattice,
        indices_xy=jnp.asarray(indices),
        products=products,
        reflection=reflection,
        domain=domain,
        checks=checks,
        memory=sc.MemoryBudget(2 * gib, gib, gib, 6 * gib),
        mirror_checks=mirror,
    )
    return arguments, check_xy, check_images, center, pitch


def main(root, out_dir):
    """Admit, read and compare a delivery, then write the figures and JSON."""
    arguments, check_xy, check_images, center, pitch = load(root)
    start = time.perf_counter()
    admission = sc.admit_grid(**arguments)
    admit_seconds = time.perf_counter() - start
    reader = Interpolator(
        admission.require(),
        arguments["domain"],
        sc.KernelProfile(),
        sc.ReadPolicy(),
        sc.Tiling(pixel_tile=2048),
        arguments["memory"],
    )
    batch = eqx.filter_jit(read_batch)
    product = jnp.zeros(len(check_xy), dtype=jnp.int32)
    reads = batch(reader, jnp.asarray(check_xy), product)
    jax.block_until_ready(reads.intensity)
    start = time.perf_counter()
    reads = batch(reader, jnp.asarray(check_xy), product)
    jax.block_until_ready(reads.intensity)
    read_seconds = (time.perf_counter() - start) / len(check_xy)
    images = np.asarray(reads.intensity)
    peaks = np.array(
        [_peak(check_images[i], xy, center, pitch) for i, xy in enumerate(check_xy)]
    )
    errors = np.abs(images - check_images).max(axis=(1, 2)) / peaks
    # Each pixel's brightest stored value over the quadrant and its mirrors:
    # the scale the kernel's truncation error follows.
    stored = np.asarray(arguments["intensity_nodes"][0]).max(axis=0)
    brightest = np.maximum.reduce(
        [stored, stored[:, ::-1], stored[::-1, :], stored[::-1, ::-1]]
    )
    pixel_relative = (np.abs(images - check_images) / brightest).max(axis=(1, 2))
    blend_path = root / "reads.npz"
    blend = np.load(blend_path)["yippy"] if blend_path.exists() else None
    blend_errors = (
        np.abs(blend - check_images).max(axis=(1, 2)) / peaks
        if blend is not None
        else None
    )
    receipt = admission.receipt
    bases = {}
    for row in _rows(root / "starter_mirror_checks.csv"):
        key = (int(row["base_ix"]), int(row["base_iy"]))
        bases[key] = bases.get(key, 0) + 1
    both = {k: v for k, v in bases.items() if k[0] != 0 and k[1] != 0}
    axis = {k: v for k, v in bases.items() if k[0] == 0 or k[1] == 0}
    summary = dict(
        mirror_bases_both_indices_nonzero=len(both),
        mirror_checks_at_those_bases=sum(both.values()),
        mirror_bases_on_an_axis=len(axis),
        mirror_checks_at_axis_bases=sum(axis.values()),
        axis_base_indices=[list(k) for k in axis],
        admitted=bool(admission.admitted),
        training_nodes=int(arguments["indices_xy"].shape[0]),
        independent_checks=len(check_xy),
        mirror_checks=len(receipt.mirror_local_peak),
        admission_seconds=round(admit_seconds, 1),
        seconds_per_read=round(read_seconds, 3),
        worst_check_error_local_peak=float(errors.max()),
        worst_check_error_pixel_brightest=float(pixel_relative.max()),
        mirror_residual_max=float(max(receipt.mirror_local_peak)),
        axis_residual_max=float(np.nanmax(np.asarray(receipt.axis_relative, float))),
        blend_worst_error_local_peak=(
            float(blend_errors.max()) if blend_errors is not None else None
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")

    hwostyle.use("dark")
    worst = int(np.argmax(errors))
    source = check_xy[worst]
    pixels = arguments["products"].pixels
    columns = 4 if blend is not None else 3
    fig, axes = plt.subplots(1, columns, figsize=(3.3 * columns, 3.4))
    viz.plot_read(check_images[worst], pixels=pixels, ax=axes[0])
    viz.plot_read(images[worst], pixels=pixels, ax=axes[1])
    viz.plot_residual(
        images[worst],
        check_images[worst],
        source_xy_lod=source,
        pixels=pixels,
        ax=axes[2],
    )
    titles = ["direct propagation", "yippy.sinc read", "yippy.sinc - direct"]
    if blend is not None:
        viz.plot_residual(
            blend[worst],
            check_images[worst],
            source_xy_lod=source,
            pixels=pixels,
            ax=axes[3],
        )
        titles.append("blend - direct")
    for ax, title in zip(axes, titles, strict=True):
        ax.set_xlim(source[0] - 3, source[0] + 3)
        ax.set_ylim(source[1] - 3, source[1] + 3)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "worst_check.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 2.8), constrained_layout=True)
    viz.plot_check_residuals(receipt, independent=errors, ax=ax)
    fig.savefig(out_dir / "check_residuals.png", dpi=130)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    main(Path(sys.argv[1]), here.parent / "_static" / "validation")
