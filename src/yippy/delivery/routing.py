"""Construction-time selection of the optional sampled-intensity backend."""

import json
from dataclasses import asdict

import jax

from yippy._version import __version__
from yippy.sinc.evidence import (
    Admission,
    AdmissionRefused,
    MemoryLimit,
    PrecisionUnavailable,
    Reason,
    Receipt,
    reason_names,
)
from yippy.sinc.reader import ReadPolicy, make_reader

from ..interpolation import InterpolationInfo, SincOptions
from .backend import _FALLBACK_MASK, DeliveryBackend
from .format import (
    admit_delivery,
    campaign_sampling_failed,
    delivery_memory,
    delivery_profile,
    detect_delivery,
    load_delivery,
)


def route_interpolation(path, requested, options, downsample_shape, *, package=None):
    """Select and validate the off-axis route before legacy data loading.

    Args:
        path: Package directory.
        requested: Requested mode: ``"blend"``, ``"auto"`` or ``"sinc"``.
        options: SincOptions of the request; None uses the defaults.
        downsample_shape: Requested (ny, nx) downsampled PSF shape; it is refused for a
            versioned delivery.
        package: Open package supplying the manifest and arrays; None reads them from
            ``path``.

    Returns:
        Tuple of the validated delivery (None for a legacy package), the delivery
        backend (None for a legacy package) and the routing information.
    """
    if requested not in ("blend", "auto", "sinc"):
        raise ValueError("interpolation must be 'blend', 'auto', or 'sinc'")
    is_blend = requested == "blend"
    if options is None:
        options = SincOptions()
    if not isinstance(options, SincOptions):
        raise TypeError("sinc_options must be SincOptions")
    if is_blend and options.policy is not None and options.policy.mode == "bounded":
        raise ValueError("Bounded policies require sinc; blend has no error bound")
    manifest = package._manifest if package else detect_delivery(path)
    if manifest is not None and is_blend and manifest["layout"] == "quadrant_rectangle":
        raise ValueError(
            "blend is unavailable for a quadrant delivery: "
            "no midpoint-aware blend representation exists"
        )
    if manifest is None and is_blend:
        return None, None, InterpolationInfo(requested, "blend")
    canonical_options = _options_identity(options)
    # Complete all file, declaration and integrity checks before any route gate.
    delivery = None
    if manifest is not None:
        if downsample_shape is not None:
            raise ValueError("Downsampling versioned sinc deliveries is not supported")
        delivery = load_delivery(path, manifest, options, package=package)
        if campaign_sampling_failed(delivery):
            raise ValueError(
                "Failed campaign sampling claim must be removed by a revised manifest"
            )
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
                ("LAYOUT_UNSUPPORTED",),
                options_json=canonical_options,
            ),
        )
    base = dict(
        manifest_sha256=delivery.manifest_sha256,
        product_id=delivery.plane["id"],
        manifest_json=json.dumps(manifest, sort_keys=True),
        options_json=canonical_options,
    )
    if not is_blend:
        base.update(
            yippy_version=__version__,
            kernel_profile=delivery_profile(delivery, options).identity,
        )

    def blend_result(reasons=(), bits=0):
        if options.policy is not None and options.policy.mode == "bounded":
            raise ValueError("Bounded policies require sinc; blend has no error bound")
        if manifest["layout"] == "quadrant_rectangle":
            raise ValueError("Quadrant deliveries cannot fall back to blend")
        return (
            delivery,
            DeliveryBackend(delivery, None, requested, bits),
            InterpolationInfo(requested, "blend", reasons, **base),
        )

    if is_blend:
        return blend_result()
    if delivery.missing_metadata:
        if (
            requested != "auto"
            or manifest["layout"] == "quadrant_rectangle"
            or (options.policy is not None and options.policy.mode != "sampling")
        ):
            raise AdmissionRefused(
                Admission(None, Receipt(), int(Reason.METADATA_MISSING))
            )
        return blend_result(("METADATA_MISSING",), int(Reason.METADATA_MISSING))
    try:
        delivery_memory(
            delivery,
            options.memory,
            profile=delivery_profile(delivery, options),
        )
    except MemoryLimit:
        if (
            requested != "auto"
            or manifest["layout"] == "quadrant_rectangle"
            or (options.policy is not None and options.policy.mode != "sampling")
        ):
            raise
        return blend_result(("MEMORY_LIMIT",), int(Reason.MEMORY_LIMIT))
    if not jax.config.x64_enabled:
        if (
            requested != "auto"
            or manifest["layout"] == "quadrant_rectangle"
            or (options.policy is not None and options.policy.mode != "sampling")
        ):
            raise PrecisionUnavailable("Enable jax_enable_x64 before sinc construction")
        return blend_result(
            ("PRECISION_UNAVAILABLE",), int(Reason.PRECISION_UNAVAILABLE)
        )
    admission, domain, memory = admit_delivery(delivery, options)
    base.update(receipt_id=admission.receipt.identity, receipt=admission.receipt)
    if not admission.admitted:
        if (
            requested != "auto"
            or manifest["layout"] == "quadrant_rectangle"
            or (options.policy is not None and options.policy.mode != "sampling")
            or admission.reason_bits & ~_FALLBACK_MASK
        ):
            admission.require()
        return blend_result(reason_names(admission.reason_bits), admission.reason_bits)
    policy = options.policy or ReadPolicy()
    if policy.mode == "bounded":
        policy = ReadPolicy("bounded", policy.max_abs_error / delivery.conversion)
    reader = make_reader(
        admission.require(),
        domain,
        profile=delivery_profile(delivery, options),
        policy=policy,
        memory=memory,
    )
    return (
        delivery,
        DeliveryBackend(delivery, reader, requested),
        InterpolationInfo(requested, "sinc", **base),
    )


def cache_interpolation_identity(info, backend):
    """Bind routing, declarations, policy and dynamic reader receipt to caches.

    Args:
        info: Routing information of the product.
        backend: Off-axis backend; a delivery backend with a reader adds its policy,
            domain, profile and receipt.

    Returns:
        Dictionary of the values that identify the interpolation route in a cache key.
    """
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
        "taps": options.taps,
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
