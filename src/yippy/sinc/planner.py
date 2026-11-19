"""Tap-count planning for prefiltered reads from delivery metadata alone."""

import math
from dataclasses import dataclass

import numpy as np

from yippy.sinc.admission import plan_spacing, sampling_ratio, sampling_reasons
from yippy.sinc.evidence import PlanRefused, Reason
from yippy.sinc.geometry import spacing_xy
from yippy.sinc.kernel import KernelProfile, PrefilteredKernelProfile
from yippy.sinc.prefilter import design_grid, design_record, sampling_ratios

_TAP_RANGE = range(5, 11)


@dataclass(frozen=True)
class PrefilterPlan:
    """Host result of ``plan_prefilter``; call require() for the profile.

    ``tone_error_two_axis`` is the estimated worst two-axis tone error of the
    chosen profile (NaN on refusal), ``kernel_design`` its full design record,
    and ``candidates`` the (taps, two-axis tone error) pairs that were tried, in
    order. These are sampled maxima of the kernel's response on the declared
    band, not error bounds on a delivered image.
    """

    profile: PrefilteredKernelProfile | None
    tone_error_two_axis: float
    candidates: tuple[tuple[int, float], ...]
    reason_bits: int
    kernel_design: tuple[tuple[str, int | float], ...] = ()

    @property
    def planned(self):
        """Whether some tap count met the request."""
        return self.profile is not None

    def require(self) -> PrefilteredKernelProfile:
        """Return the profile, or raise PlanRefused carrying this plan."""
        if self.profile is None:
            raise PlanRefused(self)
        return self.profile


def plan_prefilter(*, declared_lattice, products, reflection, domain, max_tone_error):
    """Choose the fewest taps whose design tone error meets ``max_tone_error``.

    Only metadata is read: no samples are touched and nothing is admitted. For
    5 to 10 taps, in order, the planner designs the filter that admission would
    build for this lattice, band and domain, and stops at the first tap count
    whose estimated two-axis tone error (e_x + e_y + e_x * e_y) is at or below
    ``max_tone_error``. It never tries more than 10 taps: a refusal means the
    lattice needs the raw profile or a looser request.

    The tone error is a dimensionless property of the kernel on the declared
    band. It is not ``ReadPolicy.max_abs_error`` and does not include the
    rounding of coefficients to the storage dtype, which for float32 storage is
    of order 1e-7 of the local intensity scale.

    Args:
        declared_lattice: The delivery's source PixelGrid, as passed to ``admit_grid``.
        products: The delivery's Products; their shortest declared wavelength
            sets the design band.
        reflection: The delivery's Reflection declaration.
        domain: The Domain reads will be served in; table ends it cannot reach
            do not count against a tap count.
        max_tone_error: Largest acceptable two-axis tone error, positive.

    Returns:
        A PrefilterPlan. Refusal reasons are NYQUIST_FAIL or KERNEL_MARGIN_FAIL
        when the lattice is too coarse for any prefiltered profile,
        INCOMPLETE_STENCIL when no tap count has a covered read in the domain,
        and TOLERANCE_EXCEEDED when 10 taps do not reach the request.
    """
    if not max_tone_error > 0:
        raise ValueError("max_tone_error must be positive")
    reflected = reflection.kind != "none"
    spacing = spacing_xy(declared_lattice).tolist()
    reasons = sampling_reasons(
        (rho for _, rho in sampling_ratios(products, spacing)),
        PrefilteredKernelProfile(),
    )
    if reasons:
        return PrefilterPlan(None, float("nan"), (), int(reasons))
    candidates = []
    for taps in _TAP_RANGE:
        profile = PrefilteredKernelProfile(taps=taps)
        if min(declared_lattice.shape) < profile.minimum_size(reflected):
            continue
        record = design_record(
            design_grid(profile, products, declared_lattice, reflected, domain)
        )
        error = dict(record)["tone_error_two_axis"]
        candidates.append((taps, error))
        if error <= max_tone_error:
            return PrefilterPlan(profile, error, tuple(candidates), 0, record)
    finite = any(error < float("inf") for _, error in candidates)
    reason = Reason.TOLERANCE_EXCEEDED if finite else Reason.INCOMPLETE_STENCIL
    return PrefilterPlan(None, float("nan"), tuple(candidates), int(reason))


def plan_radial_checks(count, limit, kind, spacing, seed):
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
            raise ValueError("could not place an interior check; change the seed")
        points.append((x, y))
    return points


def plan_sampling(
    reference_nm, minimum_nm, support_extent, limits_lod, decimals=6, *, profile=None
):
    """Return spacing, ratios, served cell counts and halo-padded node counts.

    The positive quadrant serves half-open limits along both axes. The upper
    coverage margin comes from the selected kernel, including its filter halo.
    """
    profile = KernelProfile() if profile is None else profile
    if len(limits_lod) != 2 or not all(math.isfinite(v) and v > 0 for v in limits_lod):
        raise ValueError("Need two finite positive served limits")
    spacing = plan_spacing(
        reference_nm, minimum_nm, support_extent, decimals, profile=profile
    )
    ratios = tuple(
        sampling_ratio(reference_nm, minimum_nm, extent, spacing)
        for extent in support_extent
    )
    cells = tuple(math.ceil(value / spacing) for value in limits_lod)
    nodes = tuple(math.ceil(n + profile.coverage_margins[1]) for n in cells)
    return spacing, ratios, cells, nodes


def plan_mirror_bases(cells_xy, nodes_xy):
    """Choose interior, axis and outer-edge nodes for reflection checks."""
    jx, jy = (n - 1 for n in cells_xy)
    nx, ny = nodes_xy
    ordered = (
        (1, 1),
        (1, jy),
        (jx, 1),
        (jx, jy),
        (0, jy),
        (jx, 0),
        (nx - 1, 1),
        (3, ny - 1),
    )
    return tuple(dict.fromkeys(b for b in ordered if b[0] < nx and b[1] < ny))


def plan_cell_checks(cells_xy, spacing_lod, kind):
    """Place sixteen off-node checks in the outermost served box cell."""
    if kind != "box":
        raise ValueError("the cell check layout needs two limits (a box)")
    jx, jy = (n - 1 for n in cells_xy)
    return [
        (jx * spacing_lod + fx * spacing_lod, jy * spacing_lod + fy * spacing_lod)
        for fy in (0.001, 0.25, 0.625, 0.999)
        for fx in (0.001, 0.375, 0.75, 0.999)
    ]
