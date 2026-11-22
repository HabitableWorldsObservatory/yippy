"""Admission records, explicit refusal reasons, and unknown error evidence."""

from abc import abstractmethod
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from hwoutils.memory import MemoryLimit as _SizeLimit

from yippy.sinc._types import F64, I32, Boolean, check_array, check_static
from yippy.sinc.domain import Domain

if TYPE_CHECKING:
    import yippy.sinc.grid


class Reason(IntFlag):
    """Version-one uint32 reason masks; multiple reasons can apply."""

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
    """Read outcome categories, independent of reason masks."""

    REFUSED = 0
    SAMPLING_ADMITTED = 1
    BOUNDED = 2
    FALLBACK = 3


class BoundKind(IntEnum):
    """Bound targets, not an ordering of evidence strength."""

    UNKNOWN = 0
    CONDITIONAL = 1
    FINITE_MODEL = 2


class LedgerEvidence(IntEnum):
    """Evidence categories for individual error terms."""

    UNKNOWN = 0
    EMPIRICAL = 1
    DECLARED = 2
    CONDITIONAL_BOUND = 3
    VERIFIED_BOUND = 4


def reason_names(bits):
    """Decode accumulated reason masks in stable bit order."""
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
    """Opaque host result; call require() for strict construction."""

    grid: "yippy.sinc.grid.SampledGrid | None"
    receipt: Receipt
    reason_bits: int

    @property
    def admitted(self):
        """Whether a complete admitted grid is available."""
        return self.grid is not None

    def require(self) -> "yippy.sinc.grid.SampledGrid":
        """Return the grid, or raise an admission exception with this receipt."""
        if self.grid is None:
            exception = AdmissionRefused
            if self.reason_bits & Reason.PRECISION_UNAVAILABLE:
                exception = PrecisionUnavailable
            elif self.reason_bits & Reason.MEMORY_LIMIT:
                exception = MemoryLimit
            raise exception(self)
        return self.grid


class AdmissionRefused(ValueError):
    """Host construction refused, carrying its admission receipt and reasons."""

    def __init__(self, admission):
        """Attach an Admission or a standalone static resource failure."""
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


class PrecisionUnavailable(AdmissionRefused):
    """The execution configuration does not support required float64 reads."""


class PlanRefused(ValueError):
    """No prefiltered tap count meets a request; carries the plan and reasons."""

    def __init__(self, plan):
        """Attach the refused plan, its reason mask and the candidates tried."""
        self.plan = plan
        self.reason_bits = int(plan.reason_bits)
        self.candidates = tuple(plan.candidates)
        super().__init__(
            f"{', '.join(reason_names(plan.reason_bits))}: tried {self.candidates}"
        )


class ReadRefused(ValueError):
    """Eager read failure containing flattened query indices and diagnostics."""

    def __init__(self, indices, statuses, reasons):
        """Attach per-query diagnostics without retaining images."""
        self.indices = tuple(indices)
        self.statuses = tuple(statuses)
        self.reasons = tuple(reasons)
        super().__init__(
            f"Refused queries {self.indices}: statuses={self.statuses}, "
            f"reasons={tuple(reason_names(r) for r in reasons)}"
        )


class EvidenceBinding(eqx.Module):
    """Internal identity and domain binding for an error evaluator."""

    receipt_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    domain: Domain


class Ledger(eqx.Module):
    """Fixed error rows in absolute delivered max-pixel intensity units."""

    bound: F64
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
        check_array(self.bound, (size,), (jnp.float64,))
        check_array(self.evidence, (size,), (jnp.int32,))
        check_array(self.applicable, (size,), (jnp.bool_,))
        check_array(self.bound_kind, (), (jnp.int32,))


class AbstractErrorModel(eqx.Module):
    """Nominal internal seam for future audited error models."""

    binding: eqx.AbstractVar[EvidenceBinding]

    @abstractmethod
    def evaluate(
        self,
        source_xy_lod: F64,
        product_index: I32,
        weight_l1_xy: F64,
        *,
        grid: "yippy.sinc.grid.SampledGrid",
    ) -> Ledger:
        """Return a ledger bound to this grid, domain and execution profile."""


class UnknownErrorModel(AbstractErrorModel):
    """All applicable error terms remain unknown, including checked symmetry."""

    binding: EvidenceBinding

    def evaluate(
        self,
        source_xy_lod: F64,
        product_index: I32,
        weight_l1_xy: F64,
        *,
        grid: "yippy.sinc.grid.SampledGrid",
    ) -> Ledger:
        """Issue unknown rows; full signed grids have no symmetry error term."""
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
            jnp.where(applicable, jnp.inf, 0.0),
            jnp.zeros(8, dtype=jnp.int32),
            applicable,
            terms,
            self.binding.target_id,
            self.binding.metric_id,
            self.binding.profile_id,
            jnp.asarray(BoundKind.UNKNOWN, dtype=jnp.int32),
        )
