"""Construction-time selection of the optional sampled-intensity backend."""

import importlib
import json
from dataclasses import asdict

import jax

from .delivery_backend import _FALLBACK_MASK, DeliveryBackend
from .interpolation import InterpolationInfo, SincOptions
from .sinc_delivery import (
    admit_delivery,
    campaign_sampling_failed,
    delivery_memory,
    detect_delivery,
    load_delivery,
)


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
    canonical_options = _options_identity(options)
    # Complete all file, declaration and integrity checks before any route gate.
    delivery = None
    if manifest is not None:
        if downsample_shape is not None:
            raise ValueError("Downsampling versioned sinc deliveries is not supported")
        delivery = load_delivery(path, manifest, options)
        if campaign_sampling_failed(delivery):
            raise ValueError(
                "Failed campaign sampling claim must be removed by a revised manifest"
            )
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
        if options.policy is not None and options.policy.mode == "bounded":
            raise ValueError(
                "Bounded policies require sinc; legacy blend has no error bound"
            )
        return (
            None,
            None,
            InterpolationInfo(
                requested,
                "blend",
                ("EXTRA_MISSING" if missing else "LAYOUT_UNSUPPORTED",),
                options_json=canonical_options,
            ),
        )
    base = dict(
        manifest_sha256=delivery.manifest_sha256,
        product_id=delivery.plane["id"],
        manifest_json=json.dumps(manifest, sort_keys=True),
        options_json=canonical_options,
    )
    if sc is not None:
        base.update(
            sinchole_version=sc.__version__, kernel_profile=sc.KernelProfile().identity
        )

    def blend_result(reasons=(), bits=0):
        if options.policy is not None and options.policy.mode == "bounded":
            raise ValueError("Bounded policies require sinc; blend has no error bound")
        if manifest["layout"] == "quadrant_rectangle":
            if missing:
                raise SincExtraMissing(
                    "Install yippy[sinc]; quadrant deliveries cannot fall back to blend"
                )
            raise ValueError("Quadrant deliveries cannot fall back to blend")
        return (
            delivery,
            DeliveryBackend(delivery, None, requested, bits),
            InterpolationInfo(requested, "blend", reasons, **base),
        )

    if requested == "blend" or missing:
        return blend_result(
            ("EXTRA_MISSING",) if missing else (), (1 << 21) if missing else 0
        )
    if delivery.missing_metadata:
        if (
            requested != "auto"
            or manifest["layout"] == "quadrant_rectangle"
            or (options.policy is not None and options.policy.mode != "sampling")
        ):
            raise sc.AdmissionRefused(
                sc.Admission(None, sc.Receipt(), int(sc.Reason.METADATA_MISSING))
            )
        return blend_result(("METADATA_MISSING",), int(sc.Reason.METADATA_MISSING))
    try:
        delivery_memory(delivery, sc, options.memory)
    except sc.MemoryLimit:
        if (
            requested != "auto"
            or manifest["layout"] == "quadrant_rectangle"
            or (options.policy is not None and options.policy.mode != "sampling")
        ):
            raise
        return blend_result(("MEMORY_LIMIT",), int(sc.Reason.MEMORY_LIMIT))
    if not jax.config.x64_enabled:
        if (
            requested != "auto"
            or manifest["layout"] == "quadrant_rectangle"
            or (options.policy is not None and options.policy.mode != "sampling")
        ):
            raise sc.PrecisionUnavailable(
                "Enable jax_enable_x64 before sinc construction"
            )
        return blend_result(
            ("PRECISION_UNAVAILABLE",), int(sc.Reason.PRECISION_UNAVAILABLE)
        )
    admission, domain, memory = admit_delivery(delivery, options, sc)
    base.update(receipt_id=admission.receipt.identity, receipt=admission.receipt)
    if not admission.admitted:
        if (
            requested != "auto"
            or manifest["layout"] == "quadrant_rectangle"
            or (options.policy is not None and options.policy.mode != "sampling")
            or admission.reason_bits & ~_FALLBACK_MASK
        ):
            admission.require()
        return blend_result(
            sc.reason_names(admission.reason_bits), admission.reason_bits
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
    identity = {
        key: value
        for key, value in asdict(info).items()
        if key not in ("receipt", "manifest_json")
    }
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


def _options_identity(options):
    """Canonicalize every requested option even when admission cannot run."""
    domain = options.domain
    policy = options.policy
    identity = {
        "product_id": options.product_id,
        "policy": ("sampling", None)
        if policy is None or policy.mode == "sampling"
        else (policy.mode, float(policy.max_abs_error)),
        "domain": None
        if domain is None
        else (
            domain.kind,
            domain.boundary,
            domain.limits_xy_lod.tolist(),
            domain.radii_lod.tolist(),
        ),
        "memory": None
        if options.memory is None
        else tuple(
            int(getattr(options.memory, "max_" + key + "_bytes"))
            for key in ("resident", "output", "scratch", "total")
        ),
    }
    return json.dumps(identity, sort_keys=True)
