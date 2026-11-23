"""Optional interpolation configuration and photon-unit read diagnostics."""

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from .sinc import Domain, MemoryBudget, ReadPolicy
from .sinc.evidence import BoundKind, Reason, Status


@dataclass(frozen=True)
class SincOptions:
    """Select a fixed product and optional sinc construction controls.

    Attributes:
        product_id: Exact manifest plane ID, required for multiple planes.
        domain: Optional permission restriction in reference lambda/D.
        memory: Explicit MemoryBudget. None uses array-size estimates
            for admission, one query chunk, 256-pixel tiles and 1024 outputs.
            Compiler workspace and caller-owned arrays are outside this budget.
        policy: Optional ReadPolicy. Its error budget is expressed in
            incident photon fractions per pixel. None selects sampling mode.
        taps: Optional prefilter tap count (5..10); None lets the sinc package select.
    """

    product_id: str | None = None
    domain: Domain | None = None
    memory: MemoryBudget | None = None
    policy: ReadPolicy | None = None
    taps: int | None = None


@dataclass(frozen=True)
class InterpolationInfo:
    """Immutable construction routing and provenance, without image storage.

    Attributes:
        requested: Interpolation mode the caller asked for.
        effective: Interpolation mode actually used.
        reasons: Reasons the effective mode differs from the requested one.
        manifest_sha256: SHA-256 of the sinc manifest, empty when none applies.
        product_id: Manifest plane ID in use, empty when none applies.
        yippy_version: yippy version that built the routing.
        kernel_profile: Name of the sinc kernel profile.
        receipt_id: Identifier of the sinc construction receipt.
        receipt: The construction receipt object, or None.
        manifest_json: Serialized sinc manifest.
        options_json: Serialized interpolation options.
    """

    requested: str
    effective: str
    reasons: tuple[str, ...] = ()
    manifest_sha256: str = ""
    product_id: str = ""
    yippy_version: str = ""
    kernel_profile: str = ""
    receipt_id: str = ""
    receipt: Any = None
    manifest_json: str = ""
    options_json: str = ""


class ReadLedger(eqx.Module):
    """Error ledger in absolute incident photon fractions per pixel.

    Attributes:
        bound: Error bound of each ledger term, in incident photon fractions
            per pixel; infinite when unknown.
        evidence: Evidence code of each term.
        applicable: Whether each term applies to the read.
        term_ids: Names of the ledger terms.
        target_id: Identifier of the quantity the bound is expressed in.
        metric_id: Identifier of the error metric.
        profile_id: Identifier of the kernel or backend profile.
        bound_kind: Kind of bound, from the sinc ``BoundKind`` enum.
    """

    bound: jax.Array
    evidence: jax.Array
    applicable: jax.Array
    term_ids: tuple[str, ...] = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    bound_kind: jax.Array


class InterpolationRead(eqx.Module):
    """Scalar or batch image read in incident photon fractions per pixel.

    Status, reason bits and bound kinds follow the version-one sinc enums.
    selected_backend is 0 for blend and 1 for sinc. Unknown errors stay infinite.

    Attributes:
        intensity: Image read in incident photon fractions per pixel.
        accepted: Whether the read was accepted.
        sampling_ok: Whether the sampling condition held.
        status: Status code from the sinc ``Status`` enum.
        reason_bits: Bit set of ``Reason`` flags.
        bound_available: Whether an error bound is available.
        bound_kind: Kind of bound, from the sinc ``BoundKind`` enum.
        abs_error_bound: Absolute error bound in incident photon fractions per
            pixel; infinite when unknown.
        empirical_max_abs: Empirical maximum absolute error, NaN when not measured.
        ledger: Per-term error ledger.
        selected_backend: 0 for blend, 1 for sinc.
    """

    intensity: jax.Array
    accepted: jax.Array
    sampling_ok: jax.Array
    status: jax.Array
    reason_bits: jax.Array
    bound_available: jax.Array
    bound_kind: jax.Array
    abs_error_bound: jax.Array
    empirical_max_abs: jax.Array
    ledger: ReadLedger
    selected_backend: jax.Array


def blend_read(
    image,
    finite,
    reasons=0,
    *,
    fallback=False,
    target_id="incident_photon_per_pixel",
    profile_id="blend",
):
    """Attach unknown-error diagnostics.

    Args:
        image: Image read in incident photon fractions per pixel.
        finite: Whether the query position was finite.
        reasons: Initial reason bits.
        fallback: Whether a fallback path produced the image.
        target_id: Identifier of the quantity the image is expressed in.
        profile_id: Identifier of the backend profile.

    Returns:
        Read with the image set to NaN where invalid and the error bound unknown.
    """
    valid = finite & jnp.all(jnp.isfinite(image))
    # Version-one sinc reason bits: NONFINITE_QUERY and FALLBACK_USED.
    bits = jnp.asarray(reasons, jnp.uint32)
    bits = bits | jnp.where(
        valid & fallback, jnp.uint32(Reason.FALLBACK_USED), jnp.uint32(0)
    )
    bits = bits | jnp.where(finite, jnp.uint32(0), jnp.uint32(Reason.NONFINITE_QUERY))
    return InterpolationRead(
        jnp.where(valid, image, jnp.nan),
        valid,
        jnp.asarray(False),
        jnp.where(valid, jnp.int32(Status.FALLBACK), jnp.int32(Status.REFUSED)),
        bits,
        jnp.asarray(False),
        jnp.int32(BoundKind.UNKNOWN),
        jnp.asarray(jnp.inf),
        jnp.asarray(jnp.nan),
        ReadLedger(
            jnp.array([jnp.inf]),
            jnp.zeros(1, jnp.int32),
            jnp.ones(1, bool),
            ("blend_error",),
            target_id,
            "absolute_max_pixel",
            profile_id,
            jnp.int32(BoundKind.UNKNOWN),
        ),
        jnp.int32(0),
    )
