"""Admission records, explicit refusal reasons, and unknown error evidence."""

from abc import abstractmethod
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from hwoutils.memory import MemoryLimit as _SizeLimit

from yippy.sinc._types import (
    FLOAT_DTYPES,
    I32,
    Boolean,
    RealArray,
    check_array,
    check_static,
)
from yippy.sinc.domain import Domain

if TYPE_CHECKING:
    import yippy.sinc.grid


class Reason(IntFlag):
    """Version-one uint32 reason masks; multiple reasons can apply.

    Attributes:
        LAYOUT_UNSUPPORTED: Source layout or numeric range is not supported.
        GRID_INCOMPLETE: Stored nodes do not fill the declared rectangle.
        GRID_DUPLICATE: Stored node indices repeat.
        GRID_NONUNIFORM: Source nodes are not a uniform axis-aligned lattice.
        DATA_INVALID: Samples or independent checks are invalid.
        METADATA_MISSING: Required declaration is absent.
        METADATA_CONFLICT: Declarations contradict each other.
        NORMALIZATION_UNSUPPORTED: Normalization reference is not supported.
        FRAME_UNSUPPORTED: Coordinate frame is not supported.
        UNKNOWN_BANDWIDTH: Support bandwidth is not declared.
        NYQUIST_FAIL: Sampling ratio is not below the Nyquist limit.
        KERNEL_MARGIN_FAIL: Sampling ratio exceeds the kernel profile margin.
        INCOMPLETE_STENCIL: A kernel stencil would leave the stored table.
        OUTSIDE_DOMAIN: Query lies outside the permitted domain.
        SYMMETRY_UNSUPPORTED: Declared symmetry is not supported.
        PRECISION_UNAVAILABLE: Required float64 execution is unavailable.
        MEMORY_LIMIT: Static allocation plan exceeds the memory budget.
        NONFINITE_QUERY: Query coordinate is not finite.
        PRODUCT_UNAVAILABLE: Requested product is not available.
        BOUND_UNKNOWN: No error bound is known.
        TOLERANCE_EXCEEDED: Error exceeds the requested tolerance.
        EXTRA_MISSING: A required optional dependency is missing.
        FALLBACK_USED: A fallback evaluation path was used.
        EMPIRICAL_ONLY: Evidence is empirical rather than a bound.
        SYMMETRY_CHECK_FAILED: A symmetry check did not pass.
        SYMMETRY_CHECK_MISSING: A required symmetry check was not supplied.
        SYMMETRY_REFERENCE_INVALID: The symmetry check reference is invalid.
    """

    LAYOUT_UNSUPPORTED = 1 << 0
    GRID_INCOMPLETE = 1 << 1
    GRID_DUPLICATE = 1 << 2
    GRID_NONUNIFORM = 1 << 3
    DATA_INVALID = 1 << 4
    METADATA_MISSING = 1 << 5
    METADATA_CONFLICT = 1 << 6
    NORMALIZATION_UNSUPPORTED = 1 << 7
    FRAME_UNSUPPORTED = 1 << 8
    UNKNOWN_BANDWIDTH = 1 << 9
    NYQUIST_FAIL = 1 << 10
    KERNEL_MARGIN_FAIL = 1 << 11
    INCOMPLETE_STENCIL = 1 << 12
    OUTSIDE_DOMAIN = 1 << 13
    SYMMETRY_UNSUPPORTED = 1 << 14
    PRECISION_UNAVAILABLE = 1 << 15
    MEMORY_LIMIT = 1 << 16
    NONFINITE_QUERY = 1 << 17
    PRODUCT_UNAVAILABLE = 1 << 18
    BOUND_UNKNOWN = 1 << 19
    TOLERANCE_EXCEEDED = 1 << 20
    EXTRA_MISSING = 1 << 21
    FALLBACK_USED = 1 << 22
    EMPIRICAL_ONLY = 1 << 23
    SYMMETRY_CHECK_FAILED = 1 << 24
    SYMMETRY_CHECK_MISSING = 1 << 25
    SYMMETRY_REFERENCE_INVALID = 1 << 26


class Status(IntEnum):
    """Read outcome categories, independent of reason masks.

    Attributes:
        REFUSED: The read was refused.
        SAMPLING_ADMITTED: The read was admitted on sampling grounds.
        BOUNDED: The read carries an error bound.
        FALLBACK: The read used a fallback path.
    """

    REFUSED = 0
    SAMPLING_ADMITTED = 1
    BOUNDED = 2
    FALLBACK = 3


class BoundKind(IntEnum):
    """Bound targets, not an ordering of evidence strength.

    Attributes:
        UNKNOWN: No bound target is known.
        CONDITIONAL: Bound holds under the delivery's premises.
        FINITE_MODEL: Bound targets the finite sampled model.
    """

    UNKNOWN = 0
    CONDITIONAL = 1
    FINITE_MODEL = 2


class LedgerEvidence(IntEnum):
    """Evidence categories for individual error terms.

    Attributes:
        UNKNOWN: The term is unknown.
        EMPIRICAL: The term rests on empirical checks.
        DECLARED: The term is declared by the delivery.
        CONDITIONAL_BOUND: The term has a bound conditional on premises.
        VERIFIED_BOUND: The term has a verified bound.
    """

    UNKNOWN = 0
    EMPIRICAL = 1
    DECLARED = 2
    CONDITIONAL_BOUND = 3
    VERIFIED_BOUND = 4


def reason_names(bits):
    """Decode accumulated reason masks in stable bit order.

    Args:
        bits: Integer reason mask.

    Returns:
        Tuple of the names of the reasons set in the mask, in bit order.
    """
    return tuple(reason.name for reason in Reason if int(bits) & int(reason))


@dataclass(frozen=True)
class Receipt:
    """Opaque immutable host evidence; no resident image cube is retained.

    Allocation estimates include native staging and canonical reorder copies.
    Compiler workspace and externally owned tables are not measurable here.
    axis_exact records native-value flip identity independently of axis_relative,
    whose entries are NaN without a positive local reference. An axis passes
    either by exact identity or by a defined relative residual within tolerance.
    provenance retains ordered (key, text) pairs verbatim and is never traced.
    kernel_design holds ordered (key, value) pairs for a prefiltered profile and
    is empty for the raw profile. Its tone errors are sampled maxima of the
    kernel's response to in-band tones on the declared band: design figures
    under the delivery's premises, not error bounds on a delivered image.

    Attributes:
        identity: Identity string of the admitted delivery.
        schema_version: Receipt schema identifier.
        profile_id: Identifier of the kernel profile.
        floor_rule: Name of the index floor rule.
        execution_profile: Identifier of the execution configuration.
        offset_dtype: Name of the precision the source offsets were stored in.
        offset_tolerance_xy: Per-axis offset tolerance (x, y), in source
            coordinate units.
        canonicalization_residual_xy: Per-axis worst residual between delivered
            and canonical coordinates (x, y), in source coordinate units.
        stored_bounds_xy: Bounds of the stored nodes along each axis.
        served_geometry: Description of the served geometry.
        premise_evidence: Recorded evidence for the physical premises.
        provenance_ids: Identifiers of the provenance records.
        allocation_estimate: Pairs of (name, size) for the estimated host
            allocations, in bytes.
        empirical_absolute: Absolute interior check residuals.
        empirical_local_peak: Local peak references for the interior checks.
        empirical_max_abs: Largest absolute interior residual; NaN if none.
        mirror_local_peak: Relative mirror check residuals.
        axis_relative: Per-axis relative flip residuals; NaN without a positive
            local reference.
        axis_exact: Per-axis flags for exact native-value flip identity.
        provenance: Ordered (key, text) provenance pairs, kept verbatim.
        kernel_design: Ordered (key, value) design figures of a prefiltered
            profile; empty for the raw profile.
    """

    identity: str = ""
    schema_version: str = "yippy-sinc-admission-1"
    profile_id: str = ""
    floor_rule: str = "exact_quotient_floor_v1"
    execution_profile: str = ""
    offset_dtype: str = ""
    offset_tolerance_xy: tuple[float, ...] = ()
    canonicalization_residual_xy: tuple[float, ...] = ()
    stored_bounds_xy: tuple = ()
    served_geometry: tuple = ()
    premise_evidence: tuple = ()
    provenance_ids: tuple[str, ...] = ()
    allocation_estimate: tuple[tuple[str, int], ...] = ()
    empirical_absolute: tuple[float, ...] = ()
    empirical_local_peak: tuple[float, ...] = ()
    empirical_max_abs: float = float("nan")
    mirror_local_peak: tuple[float, ...] = ()
    axis_relative: tuple[float, ...] = ()
    axis_exact: tuple[bool, ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()
    kernel_design: tuple[tuple[str, int | float], ...] = ()


@dataclass(frozen=True)
class Admission:
    """Opaque host result; call require() for strict construction.

    Attributes:
        grid: Admitted sampled grid, or None when admission was refused.
        receipt: Evidence record of the admission.
        reason_bits: Integer reason mask; zero or informational when admitted.
    """

    grid: "yippy.sinc.grid.SampledGrid | None"
    receipt: Receipt
    reason_bits: int

    @property
    def admitted(self):
        """Whether a complete admitted grid is available."""
        return self.grid is not None

    def require(self) -> "yippy.sinc.grid.SampledGrid":
        """Return the grid, or raise an admission exception with this receipt.

        Returns:
            The admitted sampled grid.
        """
        if self.grid is None:
            exception = AdmissionRefused
            if self.reason_bits & Reason.MEMORY_LIMIT:
                exception = MemoryLimit
            raise exception(self)
        return self.grid


class AdmissionRefused(ValueError):
    """Host construction refused, carrying its admission receipt and reasons."""

    def __init__(self, admission):
        """Attach an Admission or a standalone static resource failure.

        Args:
            admission: Refused Admission, or a message describing a standalone
                memory or precision failure.
        """
        if isinstance(admission, Admission):
            self.admission = admission
            self.receipt = admission.receipt
            self.reason_bits = admission.reason_bits
            message = ", ".join(reason_names(admission.reason_bits))
        else:
            self.admission = None
            self.receipt = None
            self.reason_bits = int(
                Reason.MEMORY_LIMIT
                if isinstance(self, MemoryLimit)
                else Reason.PRECISION_UNAVAILABLE
            )
            message = str(admission)
        super().__init__(message)


class MemoryLimit(AdmissionRefused, _SizeLimit):
    """The requested static allocation plan exceeds the supplied budget.

    Also a ``hwoutils.memory.MemoryLimit``, so one ``except`` serves every
    library's size refusal.
    """


class PlanRefused(ValueError):
    """No prefiltered tap count meets a request; carries the plan and reasons."""

    def __init__(self, plan):
        """Attach the refused plan, its reason mask and the candidates tried.

        Args:
            plan: Refused plan object, with reason_bits and candidates fields.
        """
        self.plan = plan
        self.reason_bits = int(plan.reason_bits)
        self.candidates = tuple(plan.candidates)
        super().__init__(
            f"{', '.join(reason_names(plan.reason_bits))}: tried {self.candidates}"
        )


class ReadRefused(ValueError):
    """Eager read failure containing flattened query indices and diagnostics."""

    def __init__(self, indices, statuses, reasons):
        """Attach per-query diagnostics without retaining images.

        Args:
            indices: Flattened indices of the refused queries.
            statuses: Read status of each refused query.
            reasons: Reason mask of each refused query.
        """
        self.indices = tuple(indices)
        self.statuses = tuple(statuses)
        self.reasons = tuple(reasons)
        super().__init__(
            f"Refused queries {self.indices}: statuses={self.statuses}, "
            f"reasons={tuple(reason_names(r) for r in reasons)}"
        )


class EvidenceBinding(eqx.Module):
    """Internal identity and domain binding for an error evaluator.

    Attributes:
        receipt_id: Identifier of the admission receipt.
        target_id: Identifier of the error target.
        metric_id: Identifier of the error metric.
        profile_id: Identifier of the kernel profile.
        domain: Permission domain the evaluator is bound to.
    """

    receipt_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    domain: Domain


class Ledger(eqx.Module):
    """Fixed error rows in absolute delivered max-pixel intensity units.

    Attributes:
        bound: Error bound per term, shape (n_terms,), in delivered max-pixel
            intensity units.
        evidence: LedgerEvidence category per term, shape (n_terms,).
        applicable: Whether each term applies, shape (n_terms,).
        term_ids: Names of the terms, length n_terms.
        target_id: Identifier of the error target.
        metric_id: Identifier of the error metric.
        profile_id: Identifier of the kernel profile.
        bound_kind: BoundKind of the whole ledger, scalar.
    """

    bound: RealArray
    evidence: I32
    applicable: Boolean
    term_ids: tuple[str, ...] = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    bound_kind: I32

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        size = len(self.term_ids)
        check_array(self.bound, (size,), FLOAT_DTYPES)
        check_array(self.evidence, (size,), (jnp.int32,))
        check_array(self.applicable, (size,), (jnp.bool_,))
        check_array(self.bound_kind, (), (jnp.int32,))


class AbstractErrorModel(eqx.Module):
    """Nominal internal seam for future audited error models.

    Attributes:
        binding: Identity and domain binding of the model.
    """

    binding: eqx.AbstractVar[EvidenceBinding]

    @abstractmethod
    def evaluate(
        self,
        source_xy_lod: RealArray,
        product_index: I32,
        weight_l1_xy: RealArray,
        *,
        grid: "yippy.sinc.grid.SampledGrid",
    ) -> Ledger:
        """Return a ledger bound to this grid, domain and execution profile.

        Args:
            source_xy_lod: Query source coordinate (x, y), in lambda/D.
            product_index: Index of the queried product.
            weight_l1_xy: Per-axis L1 norms of the interpolation weights
                (x, y), dimensionless.
            grid: Admitted sampled grid being read.

        Returns:
            Ledger of error terms for this query.
        """


class UnknownErrorModel(AbstractErrorModel):
    """All applicable error terms remain unknown, including checked symmetry.

    Attributes:
        binding: Identity and domain binding of the model.
    """

    binding: EvidenceBinding

    def evaluate(
        self,
        source_xy_lod: RealArray,
        product_index: I32,
        weight_l1_xy: RealArray,
        *,
        grid: "yippy.sinc.grid.SampledGrid",
    ) -> Ledger:
        """Issue unknown rows; full signed grids have no symmetry error term.

        Args:
            source_xy_lod: Query source coordinate (x, y), in lambda/D.
            product_index: Index of the queried product.
            weight_l1_xy: Per-axis L1 norms of the interpolation weights
                (x, y), dimensionless.
            grid: Admitted sampled grid being read.

        Returns:
            Ledger with infinite bounds on applicable terms and zero elsewhere.
        """
        terms = (
            "source_window",
            "node_production",
            "storage",
            "coordinates",
            "weights",
            "accumulation",
            "normalization",
            "symmetry",
        )
        applicable = (
            jnp.ones(8, dtype=jnp.bool_).at[-1].set(grid.reflection.kind != "none")
        )
        return Ledger(
            jnp.where(
                applicable,
                jnp.full((), jnp.inf, source_xy_lod.dtype),
                jnp.zeros((), source_xy_lod.dtype),
            ),
            jnp.zeros(8, dtype=jnp.int32),
            applicable,
            terms,
            self.binding.target_id,
            self.binding.metric_id,
            self.binding.profile_id,
            jnp.asarray(BoundKind.UNKNOWN, dtype=jnp.int32),
        )
