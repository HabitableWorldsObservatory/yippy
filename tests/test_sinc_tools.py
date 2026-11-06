"""Reference-page example, validate and plan tests for sinc deliveries."""

import csv
import hashlib
import json
import math
import re
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from yippy import datasets
from yippy.interpolation import SincOptions
from yippy.sinc_delivery import detect_delivery, load_delivery
from yippy.sinc_tools import (
    ACCEPTANCE_LEVEL,
    SincToolsError,
    main,
    plan,
    plan_spacing,
    validate,
    write_plan,
)

DOCS = Path(__file__).parents[1] / "docs" / "sinc-delivery.md"
DATA = Path(__file__).parent / "data"

#: Mirror bases of the designer request, with its last base on row 26.
STARTER_BASES = [
    (1, 1),
    (1, 14),
    (10, 1),
    (10, 14),
    (0, 14),
    (10, 0),
    (26, 1),
    (3, 26),
]


def example_manifest_text():
    """Return the annotated example of the reference page as plain JSON text."""
    text = DOCS.read_text()
    block = re.search(r"```javascript\n(.*?)\n```", text, re.DOTALL).group(1)
    return "\n".join(re.sub(r"\s//.*$", "", line) for line in block.splitlines())


def source_images(coords):
    """Band-limited, mirror-even analytic images on the 4 by 4 example detector."""
    rows, columns = np.mgrid[:4, :4]
    x, y = (columns - 1.5) * 0.25, (rows - 1.5) * 0.25
    return np.array(
        [
            (1 + 0.2 * np.cos(2 * np.pi * 0.2 * (x - qx)))
            * (1 + 0.1 * np.cos(2 * np.pi * 0.3 * (y - qy)))
            * 1e-3
            for qx, qy in coords
        ],
        np.float32,
    )


def local_header():
    """FITS header of the example images, with units in the comments."""
    header = fits.Header()
    header["LAMBDA"] = (1e-6, "angular reference wavelength, m")
    header["MINLAM"] = (1e-6, "monochromatic wavelength, m")
    header["MAXLAM"] = (1e-6, "monochromatic wavelength, m")
    header["N_LAM"] = 1
    header["D"] = (15.0, "entrance diameter, m")
    header["PIXSCALE"] = (0.25, "lambda0/D per pixel")
    header["XCENTER"] = 1.5
    header["YCENTER"] = 1.5
    header["BUNIT"] = "incident_fraction_per_pixel"
    return header


def build_example_delivery(root):
    """Write the small files the documented example describes and fill its hashes."""
    manifest = json.loads(example_manifest_text())
    h = manifest["source_spacing_lod"][0]
    axis = np.arange(18)
    ix, iy = np.meshgrid(axis, axis, indexing="xy")
    indices = np.column_stack((ix.ravel(), iy.ravel()))
    offsets = indices * h
    (root / "mono-000").mkdir(parents=True)
    (root / "checks").mkdir()
    (root / "mirror-checks").mkdir()
    fits.writeto(
        root / "mono-000/offax_psf.fits", source_images(offsets), local_header()
    )
    fits.writeto(root / "mono-000/offax_psf_offset_list.fits", offsets)
    with (root / "starter_training.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "ix", "iy", "x_lod", "y_lod"])
        for index, (k, q) in enumerate(zip(indices, offsets, strict=True)):
            writer.writerow([index, *k, *q])
    for name, key in (("mirror-checks", "mirror_checks"), ("checks", "checks")):
        coords = [record["source_lod_xy"] for record in manifest[key]]
        fits.writeto(
            root / name / "mono-000.fits", source_images(coords), local_header()
        )
    (root / "mirror-report.json").write_text('{"status": "pass"}')
    (root / "legacy").mkdir()
    digest = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()  # noqa: E731
    text = example_manifest_text()
    for name in set(re.findall(r"<sha256 of ([^>]+)>", text)):
        text = text.replace(f"<sha256 of {name}>", digest(name))
    (root / "sinc-delivery.json").write_text(text)
    return json.loads(text)


@pytest.fixture(scope="module")
def example_root(tmp_path_factory):
    """One small delivery built from the documented example, shared by tests."""
    root = tmp_path_factory.mktemp("sinc-example")
    build_example_delivery(root)
    return root


def test_documented_example_parses(example_root):
    """Fails if the reference page example drifts from what the parser accepts."""
    manifest = detect_delivery(example_root)
    delivery = load_delivery(example_root, manifest, SincOptions())
    assert delivery.missing_metadata == ()
    assert delivery.shape_yx == (18, 18)
    assert delivery.nodes.shape == (324, 4, 4)
    assert len(manifest["mirror_checks"]) == 5


def test_validate_passes_documented_example(example_root):
    """A delivery built from the documented example passes with its residuals."""
    pytest.importorskip("sinchole")
    report = validate(example_root)
    assert report.passed, report.summary()
    assert report.admitted
    assert report.reasons == ()
    assert report.receipt_id
    assert report.rho == pytest.approx((0.76, 0.76))
    assert len(report.interior_residuals) == 2
    assert report.interior_worst <= ACCEPTANCE_LEVEL
    assert len(report.mirror_residuals) == 5
    assert report.mirror_worst <= 1e-6
    assert "PASS" in report.summary()


def test_validate_cli_exit_codes(example_root, tmp_path, capsys):
    """The command exits 0 on a pass and 1 on a failure, printing the summary."""
    pytest.importorskip("sinchole")
    assert main(["validate", str(example_root)]) == 0
    assert "PASS" in capsys.readouterr().out
    broken = tmp_path / "broken"
    shutil.copytree(example_root, broken)
    path = broken / "mono-000/offax_psf.fits"
    data = bytearray(path.read_bytes())
    data[-5] ^= 0xFF
    path.write_bytes(bytes(data))
    assert main(["validate", str(broken)]) == 1
    assert "sha256 mismatch" in capsys.readouterr().out


def test_validate_fails_missing_mirror_checks(example_root, tmp_path):
    """Dropping the negative-position checks is refused by admission."""
    pytest.importorskip("sinchole")
    broken = tmp_path / "no-mirror"
    shutil.copytree(example_root, broken)
    manifest = json.loads((broken / "sinc-delivery.json").read_text())
    manifest["mirror_checks"] = []
    (broken / "sinc-delivery.json").write_text(json.dumps(manifest))
    report = validate(broken)
    assert not report.passed
    assert "SYMMETRY_CHECK_MISSING" in report.reasons


def test_validate_reports_unrecognized_directory(tmp_path):
    """A directory without a manifest is a usage error, not a pass or a crash."""
    pytest.importorskip("sinchole")
    with pytest.raises(SincToolsError):
        validate(tmp_path)
    assert main(["validate", str(tmp_path)]) == 2


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def rows_of(path):
    """Read a CSV file into a list of row dicts."""
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_plan_spacing_follows_the_occupancy_limit():
    """The spacing is the largest rounded value with rho at or below 0.76."""
    assert plan_spacing(1e-6, 1e-6, (1.0, 1.0)) == 0.38
    for lambda_min, support in ((1e-6, (0.982, 0.946)), (8e-7, (1.0, 1.0))):
        h = plan_spacing(1e-6, lambda_min, support)
        sigma = 1e-6 / lambda_min
        assert 2 * sigma * max(support) * h <= 0.76
        assert 2 * sigma * max(support) * (h + 1e-6) > 0.76


def test_plan_reproduces_the_starter_lists(tmp_path):
    """The training list, cell checks and request mirror bases match the starter."""
    result = plan(
        1e-6,
        1e-6,
        1.0,
        (4.18, 5.7),
        check_layout="cell",
        mirror_bases=STARTER_BASES,
    )
    assert result.spacing_lod == 0.38
    assert result.nodes_xy == (27, 31)
    paths = write_plan(result, tmp_path)
    for name in ("starter_training", "starter_checks", "starter_mirror_checks"):
        written = paths[name + ".csv"].read_text()
        assert written == (DATA / (name + ".csv")).read_text(), name
    assert len(result.training) == 837
    assert len(result.checks) == 16
    assert len(result.mirror_checks) == 20


def test_plan_default_rules(tmp_path):
    """Default plan: true outer-edge bases, 20 mirror checks, off-node checks."""
    result = plan(1e-6, 1e-6, 1.0, (4.18, 5.7), seed=3)
    nx, ny = result.nodes_xy
    assert (nx, ny) == (27, 31)
    bases = {(r["base_ix"], r["base_iy"]) for r in result.mirror_checks}
    assert (nx - 1, 1) in bases
    assert (3, ny - 1) in bases
    assert len(result.mirror_checks) == 20
    assert len(result.checks) == 16
    h = result.spacing_lod
    for row in result.checks:
        x, y = row["x_lod"], row["y_lod"]
        assert 0 < x < 4.18 and 0 < y < 5.7
        cell = np.array([x, y]) / h
        assert np.all(np.abs(cell - np.rint(cell)) > 1e-3)
    radii = np.array([math.hypot(r["x_lod"], r["y_lod"]) for r in result.checks])
    assert radii.max() / radii.min() > 3
    again = plan(1e-6, 1e-6, 1.0, (4.18, 5.7), seed=3)
    assert again.checks == result.checks
    assert plan(1e-6, 1e-6, 1.0, (4.18, 5.7), seed=4).checks != result.checks


def test_plan_single_limit_is_an_annulus(tmp_path):
    """One limit gives a circular served region and ceil(R / h) + 16 nodes."""
    result = plan(1e-6, 1e-6, 1.0, 24.72)
    assert result.nodes_xy == (82, 82)
    assert len(result.training) == 6724
    assert result.served()["kind"] == "annulus"
    assert all(math.hypot(r["x_lod"], r["y_lod"]) < 24.72 for r in result.checks)


def test_plan_cli_writes_files(tmp_path, capsys):
    """The plan subcommand writes the three lists and the manifest values."""
    code = main(
        [
            "plan",
            "--lambda0",
            "1e-6",
            "--lambda-min",
            "1e-6",
            "--support",
            "1",
            "--limit",
            "4.18",
            "5.7",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert len(rows_of(tmp_path / "starter_training.csv")) == 837
    summary = json.loads((tmp_path / "plan.json").read_text())
    assert summary["source_spacing_lod"] == [0.38, 0.38]
    assert summary["nodes_xy"] == [27, 31]
    assert (
        main(
            [
                "plan",
                "--lambda0",
                "1e-6",
                "--lambda-min",
                "2e-6",
                "--support",
                "1",
                "--limit",
                "4",
                "--out",
                str(tmp_path),
            ]
        )
        == 2
    )


def test_starter_dataset_registry_is_well_formed():
    """The starter entry pins a sha256 and a tarball URL without downloading."""
    registry = datasets._make_sinc_pikachu("unused").registry
    assert set(registry) == {datasets.SINC_STARTER_ARCHIVE}
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", registry[datasets.SINC_STARTER_ARCHIVE])
    assert datasets.SINC_STARTER_URL.startswith("https://")
    assert datasets.SINC_STARTER_URL.endswith(datasets.SINC_STARTER_ARCHIVE)


def test_unpublished_starter_raises_a_clear_error(tmp_path):
    """Without a published asset or a cached copy, the fetch says why."""
    if datasets.SINC_STARTER_PUBLISHED:
        pytest.skip("The starter delivery is published")
    with pytest.raises(RuntimeError, match="not published yet"):
        datasets.fetch_sinc_starter(tmp_path)


def test_unpublished_starter_uses_a_cached_copy(tmp_path, monkeypatch):
    """A cached archive with the pinned hash is unpacked without a download."""
    source = tmp_path / "src" / datasets.SINC_STARTER_NAME
    source.mkdir(parents=True)
    (source / "manifest.json").write_text("{}")
    cache = tmp_path / "cache"
    cache.mkdir()
    archive = cache / datasets.SINC_STARTER_ARCHIVE
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname=datasets.SINC_STARTER_NAME)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(datasets, "SINC_STARTER_SHA256", f"sha256:{digest}")
    path = Path(datasets.fetch_sinc_starter(cache))
    assert path.name == datasets.SINC_STARTER_NAME
    assert (path / "manifest.json").is_file()
