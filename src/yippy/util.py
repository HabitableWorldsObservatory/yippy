"""Utility functions for the yippy package."""

import hashlib
import json
from pathlib import Path

import astropy.io.fits as fits
import numpy as np
from hwoutils.grids import Frame


def _canonical_cache_value(value):
    """Normalize a cache identity value to a JSON-stable form.

    Real numbers of any Python or NumPy type become ``float.hex`` strings, so
    ``1``, ``1.0`` and ``np.float32(1.0)`` agree while values that differ in any
    bit of their double representation stay distinct. Booleans are kept apart
    from numbers.
    """
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value).hex()
    if isinstance(value, dict):
        return {str(k): _canonical_cache_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_cache_value(v) for v in value]
    raise TypeError(f"Unsupported cache identity value: {value!r}")


def cache_identity_json(identity: dict) -> str:
    """Deterministic serialization of a cache identity mapping."""
    return json.dumps(
        _canonical_cache_value(identity), sort_keys=True, separators=(",", ":")
    )


def cache_identity_digest(identity: dict) -> str:
    """SHA-256 hex digest of the canonical serialization of ``identity``."""
    return hashlib.sha256(cache_identity_json(identity).encode()).hexdigest()


def save_coro_performance_to_fits(
    record, filename, outdir, overwrite=True, identity=None
):
    """Atomically store the complete single-axis record as float64 columns."""
    import os
    from dataclasses import asdict

    header = fits.Header()
    header["PERFVER"] = 4
    header["APERRULE"] = type(record.rule).__name__
    header["RULEJSON"] = json.dumps(asdict(record.rule), sort_keys=True)
    header["LEAKRULE"] = record.leakage_rule
    header["FRAME"] = json.dumps(asdict(record.frame), sort_keys=True)
    header["IWA_LOD"] = record.iwa_lod if np.isfinite(record.iwa_lod) else "undefined"
    header["EXTENT"] = record.data_extent_lod
    if record.stellar_diameter is not None:
        header["STARDIAM"] = record.stellar_diameter
    if identity is not None:
        header["CACHEKEY"] = cache_identity_digest(identity)
        header["CACHEID"] = cache_identity_json(identity)
    names = (
        "separation_lod",
        "throughput",
        "core_area_lod2",
        "leakage_fraction",
        "leakage_density_lod2",
        "raw_contrast",
    )
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name=name, format="D", array=getattr(record, name))
            for name in names
        ],
        name="PERFORMANCE",
    )
    outpath = Path(outdir) / filename
    if outpath.exists() and not overwrite:
        raise FileExistsError(outpath)
    temporary = outpath.with_name(f"{outpath.name}.{os.getpid()}.tmp")
    try:
        fits.HDUList([fits.PrimaryHDU(header=header), table]).writeto(
            temporary, overwrite=True
        )
        os.replace(temporary, outpath)
    finally:
        temporary.unlink(missing_ok=True)


def read_performance_cache_key(path: Path) -> str | None:
    """Identity digest, or None for missing/unreadable/old-format records."""
    try:
        header = fits.getheader(path, 0)
        return header.get("CACHEKEY") if header.get("PERFVER") == 4 else None
    except (OSError, ValueError):
        return None


def load_coro_performance_from_fits(filename: str, indir: Path):
    """Read a complete record; legacy partial tables are not supported."""
    from .apertures import Circular, HalfMaximum, Matched
    from .performance import PerformanceCurves

    with fits.open(Path(indir) / filename) as hdus:
        header = hdus[0].header
        if header.get("PERFVER") != 4:
            raise ValueError("performance cache requires PERFVER=4")
        rule_type = {
            "Circular": Circular,
            "HalfMaximum": HalfMaximum,
            "Matched": Matched,
        }[header["APERRULE"]]
        rule = rule_type(**json.loads(header["RULEJSON"]))
        fields = {
            name: np.array(hdus["PERFORMANCE"].data[name], dtype=float)
            for name in (
                "separation_lod",
                "throughput",
                "core_area_lod2",
                "leakage_fraction",
                "leakage_density_lod2",
                "raw_contrast",
            )
        }
        iwa = header["IWA_LOD"]
        return PerformanceCurves(
            **fields,
            frame=Frame(**json.loads(header["FRAME"])),
            rule=rule,
            leakage_rule=header["LEAKRULE"],
            iwa_lod=np.nan if iwa == "undefined" else float(iwa),
            data_extent_lod=header["EXTENT"],
            stellar_diameter=header.get("STARDIAM"),
        )
