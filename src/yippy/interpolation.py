"""Optional interpolation configuration and photon-unit read diagnostics."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from sinchole import Domain, MemoryBudget, ReadPolicy


@dataclass(frozen=True)
class SincOptions:
    """Select a fixed product and optional sinchole construction controls.

    Args:
        product_id: Exact manifest plane ID, required for multiple planes.
        domain: Optional permission restriction in reference lambda/D.
        memory: Explicit sinchole MemoryBudget. None uses array-size estimates
            for admission, one query chunk, 256-pixel tiles and 1024 outputs.
            Compiler workspace and caller-owned arrays are outside this budget.
        policy: Optional sinchole ReadPolicy. Its error budget is expressed in
            incident photon fractions per pixel. None selects sampling mode.
    """

    product_id: str | None = None
    domain: "Domain | None" = None
    memory: "MemoryBudget | None" = None
    policy: "ReadPolicy | None" = None


@dataclass(frozen=True)
class InterpolationInfo:
    """Immutable construction routing and provenance, without image storage."""

    requested: str
    effective: str
    reasons: tuple[str, ...] = ()
    manifest_sha256: str = ""
    product_id: str = ""
    sinchole_version: str = ""
    kernel_profile: str = ""
    receipt_id: str = ""
    receipt: Any = None
    manifest_json: str = ""
    options_json: str = ""


class ReadLedger(eqx.Module):
    """Error ledger in absolute incident photon fractions per pixel."""

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

    Status, reason bits and bound kinds follow sinchole's version-one enums.
    selected_backend is 0 for blend and 1 for sinc. Unknown errors stay infinite.
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
    profile_id="legacy_blend",
):
    """Attach unknown-error diagnostics without importing the optional reader."""
    valid = finite & jnp.all(jnp.isfinite(image))
    # Version-one sinchole reason bits: NONFINITE_QUERY and FALLBACK_USED.
    bits = jnp.asarray(reasons, jnp.uint32)
    bits = bits | jnp.where(valid & fallback, jnp.uint32(1 << 22), jnp.uint32(0))
    bits = bits | jnp.where(finite, jnp.uint32(0), jnp.uint32(1 << 17))
    return InterpolationRead(
        jnp.where(valid, image, jnp.nan),
        valid,
        jnp.asarray(False),
        jnp.where(valid, jnp.int32(3), jnp.int32(0)),
        bits,
        jnp.asarray(False),
        jnp.int32(0),
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
            jnp.int32(0),
        ),
        jnp.int32(0),
    )
