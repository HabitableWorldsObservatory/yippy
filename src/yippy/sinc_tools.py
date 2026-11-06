"""Designer tools for sinc-delivery-0.1 directories: check a delivery, plan positions.

``validate`` runs every yippy integrity check and then sinchole admission with
JAX binary64 enabled, and reports pass or fail with the numbers behind it.
``plan`` writes the training, check and mirror-check coordinate lists a designer
propagates. Both are exposed as functions and as the ``yippy-sinc`` command.

The format itself is described in the sinc delivery reference page.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import jax
import numpy as np

from .interpolation import SincOptions
from .sinc_delivery import (
    admit_delivery,
    campaign_sampling_failed,
    detect_delivery,
    load_delivery,
)

#: Largest kernel occupancy rho = 2 (lambda0 / lambda_min) L h that sinchole admits.
MAX_OCCUPANCY = 0.76

#: Interpolation residual accepted at each interior check, as a fraction of the
#: direct local peak. A rehearsal delivery of a public coronagraph design read
#: 1.4e-6 at worst, so this level leaves a factor of about seven of headroom
#: while sitting five orders below the blend reader on the same checks.
ACCEPTANCE_LEVEL = 1e-5

#: Mirror and axis residual limit, fixed by the format (reflection tolerance).
MIRROR_TOLERANCE = 1e-6

#: Half-width of the interpolation stencil, in nodes.
STENCIL_HALF_WIDTH = 16

#: Default number of interior checks that ``plan`` spreads over radius.
DEFAULT_CHECK_COUNT = 16

#: Cell fractions of the cell layout of checks, per axis.
CELL_FRACTIONS_X = (0.001, 0.375, 0.75, 0.999)
CELL_FRACTIONS_Y = (0.001, 0.25, 0.625, 0.999)


class SincToolsError(ValueError):
    """A designer-facing input or environment problem, never a traceback."""


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationReport:
    """Outcome of ``validate``: every number the pass or fail rests on.

    Attributes:
        passed: True only if integrity, admission, interior checks, mirror
            checks and axis checks all pass.
        failures: One sentence per failed requirement, empty on a pass.
        notes: Advisory remarks that do not change the verdict.
        integrity_error: Message of the first yippy integrity failure, or None.
        reasons: sinchole reason names that refused or flagged admission.
        admitted: Whether sinchole admitted the grid.
        receipt_id: Identity of the admission receipt, or None if none exists.
        rho: Kernel occupancy per axis, x then y, or None if not computable.
        interior_residuals: Per interior check, max abs difference over local peak.
        interior_worst: Largest interior residual, or NaN.
        interior_median: Median interior residual, or NaN.
        acceptance_level: Interior residual level the verdict used.
        mirror_residuals: Per negative-position check residual over local peak.
        axis_residuals: Per axis image residual, NaN where the flip was exact.
        axis_exact: Per axis image, whether the flip is bit-for-bit identical.
        mirror_worst: Largest mirror or axis residual, or NaN.
        mirror_tolerance: Mirror and axis residual limit.
    """

    passed: bool
    failures: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    integrity_error: str | None = None
    reasons: tuple[str, ...] = ()
    admitted: bool = False
    receipt_id: str | None = None
    rho: tuple[float, float] | None = None
    interior_residuals: tuple[float, ...] = ()
    interior_worst: float = math.nan
    interior_median: float = math.nan
    acceptance_level: float = ACCEPTANCE_LEVEL
    mirror_residuals: tuple[float, ...] = ()
    axis_residuals: tuple[float, ...] = ()
    axis_exact: tuple[bool, ...] = ()
    mirror_worst: float = math.nan
    mirror_tolerance: float = MIRROR_TOLERANCE
    product_id: str | None = None

    def to_dict(self):
        """Return a JSON-serializable view of the report."""
        out = {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in self.__dict__.items()
        }
        return json.loads(json.dumps(out, default=float, allow_nan=True))

    def summary(self):
        """Return the printed pass or fail summary."""
        lines = [
            f"{'PASS' if self.passed else 'FAIL'}: sinc delivery"
            + (f" (product {self.product_id})" if self.product_id else ""),
        ]
        if self.integrity_error:
            lines.append(f"integrity: FAIL, {self.integrity_error}")
        else:
            lines.append("integrity: pass")
        reasons = ", ".join(self.reasons) if self.reasons else "none"
        lines.append(f"admission: {'admitted' if self.admitted else 'refused'}")
        lines.append(f"reasons: {reasons}")
        lines.append(f"receipt id: {self.receipt_id or 'none'}")
        if self.rho is not None:
            lines.append(
                f"rho (x, y): {self.rho[0]:.4f}, {self.rho[1]:.4f} "
                f"(limit {MAX_OCCUPANCY})"
            )
        lines.append(
            f"interior checks: {len(self.interior_residuals)}, "
            f"worst {_fmt(self.interior_worst)}, median {_fmt(self.interior_median)} "
            f"of local peak (acceptance {self.acceptance_level:.0e})"
        )
        exact = sum(self.axis_exact)
        lines.append(
            f"mirror checks: {len(self.mirror_residuals)}, worst "
            f"{_fmt(_nanmax(self.mirror_residuals))}; axis images: "
            f"{len(self.axis_exact)}, {exact} exact, worst non-exact "
            f"{_fmt(_nanmax(self.axis_residuals))} "
            f"(limit {self.mirror_tolerance:.0e})"
        )
        lines.extend(f"failure: {text}" for text in self.failures)
        lines.extend(f"note: {text}" for text in self.notes)
        return "\n".join(lines)


def _fmt(value):
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.3e}"


def _nanmax(values):
    finite = [v for v in values if math.isfinite(v)]
    return max(finite) if finite else math.nan


def _kernel_occupancy(manifest, plane):
    """Reproduce the core's binary64 operation order for rho on each axis."""
    reference = float(manifest["lambda0_m"]) * 1e9
    minima = [plane.get("sampling_min_wavelength_m")]
    campaign = manifest["spectral_sampling"].get("campaign_sampling_min_m")
    if campaign is not None:
        minima.append(campaign)
    minima = [m * 1e9 for m in minima if m is not None]
    extents = manifest.get("support_extent_D")
    if not minima or extents is None:
        return None
    sigma = reference / min(minima)
    return tuple(
        (2.0 * (sigma * float(extent))) * float(pitch)
        for extent, pitch in zip(extents, manifest["source_spacing_lod"], strict=True)
    )


def _outer_edge_notes(delivery):
    """Advise when no mirror base touches the largest stored index on an axis."""
    manifest = delivery.manifest
    if manifest["layout"] != "quadrant_rectangle":
        return []
    bases = {
        tuple(int(v) for v in delivery.indices[record["base_index"]])
        for record in manifest.get("mirror_checks", [])
        if record["plane_id"] == delivery.plane["id"]
    }
    notes = []
    for axis, name in ((0, "x"), (1, "y")):
        edge = int(delivery.indices[:, axis].max())
        if not any(base[axis] == edge for base in bases):
            notes.append(
                f"no mirror check base reaches the outer {name} node ({name}={edge}); "
                "the outer edge of the stored region is not covered by a negative check"
            )
    return notes


def validate(path, product_id=None, acceptance_level=ACCEPTANCE_LEVEL):
    """Check a sinc delivery directory the way yippy will read it.

    Runs every yippy integrity check (hashes, FITS and CSV agreement,
    declarations), then sinchole admission with JAX binary64 on, and gathers the
    interior, mirror and axis residuals from the admission receipt.

    Args:
        path: Delivery directory holding ``sinc-delivery.json``.
        product_id: Plane ID to check. Required when the manifest has several.
        acceptance_level: Largest accepted interior residual as a fraction of
            the direct local peak.

    Returns:
        A ValidationReport. Problems with the delivery are reported in it and do
        not raise.

    Raises:
        SincToolsError: If ``path`` holds no sinc delivery or sinchole is missing.
    """
    root = Path(path)
    try:
        manifest = detect_delivery(root)
    except (ValueError, json.JSONDecodeError) as error:
        return _failed(f"unreadable manifest: {error}", product_id)
    if manifest is None:
        raise SincToolsError(f"no sinc-delivery.json in {root}")
    try:
        import sinchole as sc
    except ModuleNotFoundError as error:
        if error.name != "sinchole":
            raise
        raise SincToolsError(
            "install yippy[sinc] to validate a sinc delivery"
        ) from error

    options = SincOptions(product_id=product_id)
    try:
        delivery = load_delivery(root, manifest, options)
    except (ValueError, KeyError, TypeError, OSError) as error:
        message = str(error) or type(error).__name__
        return _failed(message, product_id)
    plane = delivery.plane
    rho = _kernel_occupancy(manifest, plane)
    failures, notes = [], list(_outer_edge_notes(delivery))
    if campaign_sampling_failed(delivery):
        failures.append("campaign sampling claim fails rho <= 0.76 on an axis")
    if delivery.missing_metadata:
        failures.append("missing metadata: " + ", ".join(delivery.missing_metadata))

    receipt, reasons, admitted = None, (), False
    if not failures:
        with jax.enable_x64(True):
            try:
                admission, _, _ = admit_delivery(delivery, options, sc)
                receipt = admission.receipt
                reasons = sc.reason_names(admission.reason_bits)
                admitted = admission.admitted
            except sc.AdmissionRefused as refusal:
                receipt = refusal.receipt
                reasons = sc.reason_names(refusal.reason_bits)
    if not admitted and not failures:
        failures.append("sinchole refused admission: " + (", ".join(reasons) or "?"))

    interior = tuple(
        float(v) for v in (receipt.empirical_local_peak if receipt else ())
    )
    mirror = tuple(float(v) for v in (receipt.mirror_local_peak if receipt else ()))
    axis = tuple(float(v) for v in (receipt.axis_relative if receipt else ()))
    exact = tuple(bool(v) for v in (receipt.axis_exact if receipt else ()))
    if admitted:
        _judge(manifest, interior, mirror, axis, exact, acceptance_level, failures)
    worst_interior = max(interior, default=math.nan)
    pool = [v for v in (*mirror, *axis) if math.isfinite(v)]
    return ValidationReport(
        passed=not failures,
        failures=tuple(failures),
        notes=tuple(notes),
        reasons=tuple(reasons),
        admitted=admitted,
        receipt_id=receipt.identity if receipt and receipt.identity else None,
        rho=rho,
        interior_residuals=interior,
        interior_worst=worst_interior,
        interior_median=statistics.median(interior) if interior else math.nan,
        acceptance_level=acceptance_level,
        mirror_residuals=mirror,
        axis_residuals=axis,
        axis_exact=exact,
        mirror_worst=max(pool, default=math.nan),
        mirror_tolerance=MIRROR_TOLERANCE,
        product_id=plane["id"],
    )


def _judge(manifest, interior, mirror, axis, exact, level, failures):
    if not interior:
        failures.append("no interior checks were compared")
    elif not all(math.isfinite(v) for v in interior):
        failures.append("an interior check has no positive local peak reference")
    elif max(interior) > level:
        failures.append(
            f"worst interior residual {max(interior):.3e} exceeds {level:.0e}"
        )
    if manifest["layout"] == "quadrant_rectangle":
        if not mirror:
            failures.append("no mirror checks were compared")
        elif not all(math.isfinite(v) and v <= MIRROR_TOLERANCE for v in mirror):
            failures.append(
                f"mirror residual {_nanmax(mirror):.3e} exceeds {MIRROR_TOLERANCE:.0e}"
            )
        for value, is_exact in zip(axis, exact, strict=False):
            if not is_exact and not (
                math.isfinite(value) and value <= MIRROR_TOLERANCE
            ):
                failures.append(
                    f"axis residual {value:.3e} exceeds {MIRROR_TOLERANCE:.0e}"
                )
                break


def _failed(message, product_id):
    return ValidationReport(
        passed=False,
        failures=(f"integrity check failed: {message}",),
        integrity_error=message,
        product_id=product_id,
    )


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Plan:
    """Positions a designer propagates, chosen from the sampling rule.

    Attributes:
        lambda0_m: Reference wavelength in meters.
        lambda_min_m: Shortest wavelength used to choose spacing, in meters.
        support_extent_D: Full illuminated entrance widths along x and y over D.
        limit_lod: Half-open served limit per axis, in lambda0/D.
        limit_kind: ``annulus`` for one limit (radius), ``box`` for two.
        spacing_lod: Node spacing h, the same on both axes.
        rho: Kernel occupancy per axis at this spacing.
        nodes_xy: Stored node count per axis, ceil(limit / h) + 16.
        training: Rows (index, ix, iy, x_lod, y_lod), sorted by y then x.
        checks: Rows (index, id, x_lod, y_lod).
        mirror_checks: Rows of the mirror-check list.
        seed: Seed that fixes the interior check positions.
        check_layout: ``radius`` or ``cell``.
        notes: Advisory remarks.
    """

    lambda0_m: float
    lambda_min_m: float
    support_extent_D: tuple[float, float]
    limit_lod: tuple[float, float]
    limit_kind: str
    spacing_lod: float
    rho: tuple[float, float]
    nodes_xy: tuple[int, int]
    training: list
    checks: list
    mirror_checks: list
    seed: int
    check_layout: str
    notes: tuple[str, ...] = ()

    def served(self):
        """Return the manifest ``served`` block this plan is built for."""
        if self.limit_kind == "annulus":
            return {
                "kind": "annulus",
                "boundary": "lower_closed_upper_open",
                "radii_lod": [0.0, self.limit_lod[0]],
                "bounds_lod": None,
            }
        return {
            "kind": "box",
            "boundary": "lower_closed_upper_open",
            "bounds_lod": [[0.0, self.limit_lod[0]], [0.0, self.limit_lod[1]]],
            "radii_lod": None,
        }

    def summary_dict(self):
        """Return the lattice and served values to copy into the manifest."""
        return {
            "source_origin_lod": [0.0, 0.0],
            "source_spacing_lod": [self.spacing_lod, self.spacing_lod],
            "support_extent_D": list(self.support_extent_D),
            "served": self.served(),
            "layout": "quadrant_rectangle",
            "nodes_xy": list(self.nodes_xy),
            "training_rows": len(self.training),
            "check_rows": len(self.checks),
            "mirror_check_rows": len(self.mirror_checks),
            "rho": list(self.rho),
            "seed": self.seed,
            "check_layout": self.check_layout,
            "notes": list(self.notes),
        }


def plan_spacing(lambda0_m, lambda_min_m, support_extent_D, decimals=6):
    """Largest node spacing with rho <= 0.76, rounded down to ``decimals`` places.

    rho = 2 (lambda0 / lambda_min) L h, with L the larger entrance width over D.
    The same binary64 operation order as sinchole is used for the final test.

    Returns:
        The spacing h in lambda0/D.
    """
    if not (lambda_min_m > 0 and lambda0_m >= lambda_min_m):
        raise SincToolsError("need 0 < lambda_min <= lambda0")
    longest = max(support_extent_D)
    if not longest > 0:
        raise SincToolsError("support widths must be positive")
    sigma = (lambda0_m * 1e9) / (lambda_min_m * 1e9)
    limit = MAX_OCCUPANCY / (2.0 * sigma * longest)
    unit = Decimal(1).scaleb(-decimals)
    spacing = float((Decimal(repr(limit)) // unit) * unit)
    while not (2.0 * (sigma * longest)) * spacing <= MAX_OCCUPANCY:
        spacing = float(Decimal(repr(spacing)) - unit)
    if spacing <= 0:
        raise SincToolsError("spacing underflows the requested decimals")
    return spacing


def _mirror_rows(bases, spacing):
    rows = []
    for ix, iy in bases:
        signs = (
            [(-1, 1), (1, -1), (-1, -1)]
            if ix and iy
            else ([(1, -1)] if iy else [(-1, 1)])
        )
        for sx, sy in signs:
            rows.append(
                {
                    "index": len(rows),
                    "id": f"mirror-{len(rows):03d}",
                    "plane_id": "mono-000",
                    "base_ix": ix,
                    "base_iy": iy,
                    "source_ix": sx * ix,
                    "source_iy": sy * iy,
                    "x_lod": sx * ix * spacing,
                    "y_lod": sy * iy * spacing,
                    "flip_x": int(sx * ix < 0),
                    "flip_y": int(sy * iy < 0),
                }
            )
    return rows


def _radius_checks(count, limit, kind, spacing, seed):
    """Stratify the served radius into ``count`` shells, one off-node check each."""
    rng = np.random.default_rng(seed)
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    phase = float(rng.uniform())
    points = []
    for k in range(count):
        for _ in range(100):
            t = (k + float(rng.uniform(0.1, 0.9))) / count
            turn = (phase + golden * (k + 1)) % 1.0
            theta = (0.05 + 0.9 * turn) * math.pi / 2.0
            if kind == "annulus":
                x, y = t * limit[0] * math.cos(theta), t * limit[0] * math.sin(theta)
            else:
                x, y = t * limit[0] * math.cos(theta), t * limit[1] * math.sin(theta)
            cell = np.array([x, y]) / spacing
            off_node = np.all(np.abs(cell - np.rint(cell)) > 1e-3)
            inside = (
                x * x + y * y < limit[0] ** 2
                if kind == "annulus"
                else x < limit[0] and y < limit[1]
            )
            if off_node and inside:
                break
        else:
            raise SincToolsError("could not place an interior check; change the seed")
        points.append((x, y))
    return points


def plan(
    lambda0_m,
    lambda_min_m,
    support_extent_D,
    limit_lod,
    seed=0,
    check_count=DEFAULT_CHECK_COUNT,
    check_layout="radius",
    mirror_bases=None,
    decimals=6,
):
    """Choose spacing, nodes, interior checks and mirror checks for a delivery.

    Spacing comes from the 0.76 kernel occupancy limit, the node count per axis
    is ceil(limit / h) + 16, interior checks are ``check_count`` off-node
    positions spread over the served radius (or the 4 by 4 cell layout of the
    designer request, which needs a box), and mirror checks use eight bases that
    include the outer-edge nodes.

    Args:
        lambda0_m: Reference wavelength in meters.
        lambda_min_m: Shortest wavelength used to choose spacing, in meters.
        support_extent_D: One or two full entrance widths over D, x then y.
        limit_lod: One value (served radius) or two (box limits), half-open,
            in lambda0/D.
        seed: Seed for the interior check positions.
        check_count: Interior checks for the radius layout.
        check_layout: ``radius`` (stratified over radius) or ``cell`` (the
            4 by 4 fractions inside the outermost cell, 16 checks).
        mirror_bases: Optional explicit (ix, iy) mirror bases replacing the
            default eight.
        decimals: Decimal places the spacing is rounded down to.

    Returns:
        A Plan. Use ``write_plan`` to write the CSV files.
    """
    support = tuple(float(v) for v in np.atleast_1d(support_extent_D))
    support = support * 2 if len(support) == 1 else support
    limit = tuple(float(v) for v in np.atleast_1d(limit_lod))
    if len(support) != 2 or len(limit) not in (1, 2) or min(limit) <= 0:
        raise SincToolsError("support takes one or two values, limit one or two")
    kind = "annulus" if len(limit) == 1 else "box"
    if len(limit) == 1:
        limit = (limit[0], limit[0])
    if check_layout not in ("radius", "cell"):
        raise SincToolsError("check layout must be radius or cell")
    spacing = plan_spacing(lambda0_m, lambda_min_m, support, decimals)
    sigma = (lambda0_m * 1e9) / (lambda_min_m * 1e9)
    rho = tuple((2.0 * (sigma * extent)) * spacing for extent in support)
    cells = tuple(math.ceil(value / spacing) for value in limit)
    nx, ny = (n + STENCIL_HALF_WIDTH for n in cells)
    training = [
        {
            "index": iy * nx + ix,
            "ix": ix,
            "iy": iy,
            "x_lod": ix * spacing,
            "y_lod": iy * spacing,
        }
        for iy in range(ny)
        for ix in range(nx)
    ]
    jx, jy = cells[0] - 1, cells[1] - 1
    if check_layout == "cell":
        if kind != "box":
            raise SincToolsError("the cell check layout needs two limits (a box)")
        points = [
            (jx * spacing + fx * spacing, jy * spacing + fy * spacing)
            for fy in CELL_FRACTIONS_Y
            for fx in CELL_FRACTIONS_X
        ]
    else:
        points = _radius_checks(check_count, limit, kind, spacing, seed)
    checks = [
        {"index": i, "id": f"check-{i:03d}", "x_lod": x, "y_lod": y}
        for i, (x, y) in enumerate(points)
    ]
    if mirror_bases is None:
        ordered = [
            (1, 1),
            (1, jy),
            (jx, 1),
            (jx, jy),
            (0, jy),
            (jx, 0),
            (nx - 1, 1),
            (3, ny - 1),
        ]
        mirror_bases = list(
            dict.fromkeys(b for b in ordered if b[0] < nx and b[1] < ny)
        )
    mirrors = _mirror_rows([tuple(b) for b in mirror_bases], spacing)
    notes = []
    if max(support) > 1.0:
        notes.append("support wider than 1 D: confirm the widths with the evidence")
    return Plan(
        lambda0_m,
        lambda_min_m,
        support,
        limit,
        kind,
        spacing,
        rho,
        (nx, ny),
        training,
        checks,
        mirrors,
        seed,
        check_layout,
        tuple(notes),
    )


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plan(result, outdir):
    """Write the three CSV lists and ``plan.json`` into ``outdir``.

    Returns:
        The paths written, keyed by file name.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rows in (
        ("starter_training.csv", result.training),
        ("starter_checks.csv", result.checks),
        ("starter_mirror_checks.csv", result.mirror_checks),
    ):
        paths[name] = outdir / name
        _write_csv(paths[name], rows)
    paths["plan.json"] = outdir / "plan.json"
    paths["plan.json"].write_text(json.dumps(result.summary_dict(), indent=2) + "\n")
    return paths


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def _parser():
    parser = argparse.ArgumentParser(
        prog="yippy-sinc", description="Check and plan sinc-delivery-0.1 directories."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="check a delivery directory")
    check.add_argument("directory")
    check.add_argument("--product", default=None, help="plane ID for multi-plane")
    check.add_argument(
        "--level",
        type=float,
        default=ACCEPTANCE_LEVEL,
        help="interior residual acceptance, fraction of local peak",
    )
    check.add_argument("--json", action="store_true", help="print the report as JSON")
    make = commands.add_parser("plan", help="write the coordinate CSV lists")
    make.add_argument("--lambda0", type=float, required=True, help="meters")
    make.add_argument("--lambda-min", type=float, required=True, help="meters")
    make.add_argument(
        "--support", type=float, nargs="+", required=True, help="L (or Lx Ly), over D"
    )
    make.add_argument(
        "--limit",
        type=float,
        nargs="+",
        required=True,
        help="served limit R (radius) or Rx Ry (box), half-open, in lambda0/D",
    )
    make.add_argument("--seed", type=int, default=0)
    make.add_argument("--checks", type=int, default=DEFAULT_CHECK_COUNT)
    make.add_argument("--check-layout", choices=("radius", "cell"), default="radius")
    make.add_argument("--out", default=".", help="output directory")
    return parser


def main(argv=None):
    """Run the ``yippy-sinc`` command line and return the exit code."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            report = validate(args.directory, args.product, args.level)
            print(
                json.dumps(report.to_dict(), indent=2)
                if args.json
                else report.summary()
            )
            return 0 if report.passed else 1
        result = plan(
            args.lambda0,
            args.lambda_min,
            args.support,
            args.limit,
            args.seed,
            args.checks,
            args.check_layout,
        )
        paths = write_plan(result, args.out)
    except SincToolsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"spacing {result.spacing_lod} lambda0/D, rho {result.rho[0]:.4f}, "
        f"{result.nodes_xy[0]} x {result.nodes_xy[1]} nodes "
        f"({len(result.training)} training), {len(result.checks)} checks, "
        f"{len(result.mirror_checks)} mirror checks"
    )
    for name in paths:
        print(f"wrote {paths[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
