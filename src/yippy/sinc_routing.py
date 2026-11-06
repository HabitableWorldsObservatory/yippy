"""Construction-time selection of the optional sampled-intensity backend."""

import importlib
import json
from dataclasses import asdict, replace

import jax

from .delivery_backend import _FALLBACK_MASK, DeliveryBackend
from .interpolation import InterpolationInfo, SincOptions
from .sinc_delivery import admit_delivery, detect_delivery, load_delivery


class _DeliveryBudgetExceeded(ValueError):
    """The selected native allocation plan exceeds a caller ceiling."""


class SincExtraMissing(ImportError):
    """The requested reader requires the optional yippy[sinc] extra."""


def route_interpolation(path, requested, options, downsample_shape):
    """Select and validate the off-axis route before legacy data loading."""
    if requested not in ("blend", "auto", "sinc"):
        raise ValueError("interpolation must be 'blend', 'auto', or 'sinc'")
    if options is None:
        options = SincOptions()
    if not isinstance(options, SincOptions):
        raise TypeError("sinc_options must be SincOptions")
    if (
        requested == "blend"
        and options.policy is not None
        and options.policy.mode == "bounded"
    ):
        raise ValueError("Bounded policies require sinc; blend has no error bound")
    manifest = detect_delivery(path)
    if (
        manifest is not None
        and requested == "blend"
        and manifest["layout"] == "quadrant_rectangle"
    ):
        raise ValueError(
            "blend is unavailable for a quadrant delivery: "
            "no midpoint-aware blend representation exists"
        )
    if manifest is None and requested == "blend":
        return None, None, InterpolationInfo(requested, "blend")
    sc = None
    missing = False
    if requested != "blend":
        try:
            sc = importlib.import_module("sinchole")
        except ModuleNotFoundError as error:
            if error.name != "sinchole":
                raise
            if requested == "sinc" or (
                options.policy is not None and options.policy.mode == "bounded"
            ):
                raise SincExtraMissing(
                    "Install yippy[sinc] to use sinc interpolation"
                ) from error
            missing = True
    if manifest is None:
        if requested == "sinc":
            raise ValueError(
                "sinc requires a versioned sinc-delivery.json; "
                "unmarked legacy packages are unsupported"
            )
        return (
            None,
            None,
            InterpolationInfo(
                requested,
                "blend",
                ("EXTRA_MISSING" if missing else "LAYOUT_UNSUPPORTED",),
            ),
        )
    if downsample_shape is not None:
        raise ValueError("Downsampling versioned sinc deliveries is not supported")

    def preflight(resident, shape):
        if options.memory is None:
            return
        from sinchole.reader import estimate_memory

        estimates = estimate_memory(resident, shape, sc.Tiling(), 1, admission=True)
        for key in ("resident", "output", "scratch", "total"):
            if estimates[key] > getattr(options.memory, f"max_{key}_bytes"):
                raise _DeliveryBudgetExceeded(f"{key} estimate exceeds supplied budget")

    resource_refusal = None
    try:
        delivery = load_delivery(
            path,
            manifest,
            options,
            preflight=preflight if sc is not None else None,
        )
    except _DeliveryBudgetExceeded as error:
        if (
            requested != "auto"
            or manifest["layout"] == "quadrant_rectangle"
            or (options.policy is not None and options.policy.mode != "sampling")
        ):
            raise sc.MemoryLimit(str(error)) from error
        resource_refusal = error
        delivery = load_delivery(path, manifest, replace(options, memory=None))
    base = dict(
        manifest_sha256=delivery.manifest_sha256,
        product_id=delivery.plane["id"],
        manifest_json=json.dumps(manifest, sort_keys=True),
    )
    if requested == "blend" or missing:
        if manifest["layout"] == "quadrant_rectangle":
            raise SincExtraMissing(
                "Install yippy[sinc]; quadrant deliveries cannot fall back to blend"
            )
        reasons = ("EXTRA_MISSING",) if missing else ()
        backend = DeliveryBackend(
            delivery, None, requested, (1 << 21) if missing else 0
        )
        return delivery, backend, InterpolationInfo(requested, "blend", reasons, **base)
    if resource_refusal is not None:
        base.update(
            sinchole_version=sc.__version__, kernel_profile=sc.KernelProfile().identity
        )
        return (
            delivery,
            DeliveryBackend(delivery, None, requested, int(sc.Reason.MEMORY_LIMIT)),
            InterpolationInfo(requested, "blend", ("MEMORY_LIMIT",), **base),
        )
    if not jax.config.x64_enabled:
        if (
            requested != "auto"
            or manifest["layout"] == "quadrant_rectangle"
            or (options.policy is not None and options.policy.mode != "sampling")
        ):
            raise sc.PrecisionUnavailable(
                "Enable jax_enable_x64 before sinc construction"
            )
        base.update(
            sinchole_version=sc.__version__, kernel_profile=sc.KernelProfile().identity
        )
        return (
            delivery,
            DeliveryBackend(
                delivery, None, requested, int(sc.Reason.PRECISION_UNAVAILABLE)
            ),
            InterpolationInfo(requested, "blend", ("PRECISION_UNAVAILABLE",), **base),
        )
    admission, domain, memory = admit_delivery(delivery, options, sc)
    base.update(
        sinchole_version=sc.__version__,
        kernel_profile=sc.KernelProfile().identity,
        receipt_id=admission.receipt.identity,
        receipt=admission.receipt,
    )
    if not admission.admitted:
        if (
            requested != "auto"
            or manifest["layout"] == "quadrant_rectangle"
            or (options.policy is not None and options.policy.mode != "sampling")
            or admission.reason_bits & ~_FALLBACK_MASK
        ):
            admission.require()
        backend = DeliveryBackend(delivery, None, requested, admission.reason_bits)
        return (
            delivery,
            backend,
            InterpolationInfo(
                requested, "blend", sc.reason_names(admission.reason_bits), **base
            ),
        )
    policy = options.policy or sc.ReadPolicy()
    if policy.mode == "bounded":
        policy = sc.ReadPolicy("bounded", policy.max_abs_error / delivery.conversion)
    reader = sc.Interpolator(
        admission.require(), domain, sc.KernelProfile(), policy, sc.Tiling(), memory
    )
    return (
        delivery,
        DeliveryBackend(delivery, reader, requested),
        InterpolationInfo(requested, "sinc", **base),
    )


def cache_interpolation_identity(info, backend):
    """Bind routing, declarations, policy and dynamic reader receipt to caches."""
    identity = {key: value for key, value in asdict(info).items() if key != "receipt"}
    if isinstance(backend, DeliveryBackend) and backend.reader is not None:
        reader = backend.reader
        identity["policy"] = (reader.policy.mode, float(reader.policy.max_abs_error))
        identity["domain"] = (
            reader.domain.kind,
            reader.domain.boundary,
            reader.domain.limits_xy_lod.tolist(),
            reader.domain.radii_lod.tolist(),
        )
        identity["profile"] = reader.profile.identity
        identity["dynamic_receipt_id"] = reader.grid.receipt_id
    return identity
