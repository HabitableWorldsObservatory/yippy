"""Admission refusal, canonicalization, and independent empirical evidence."""

import math
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import yippy.sinc as sc
from yippy.sinc.evidence import Reason
from yippy.sinc.geometry import origin_xy, source_grid, spacing_xy
from yippy.sinc.grid import CheckSet, Reflection
from yippy.sinc.reader import Interpolator, read_batch

from .helpers import analytic, delivery, make_reader


def changed(data, key, where, value):
    return {**data, key: eqx.tree_at(where, data[key], value)}


@pytest.mark.parametrize(
    "case,reason",
    [
        ("duplicate", Reason.GRID_DUPLICATE),
        ("missing", Reason.GRID_INCOMPLETE),
        ("offset", Reason.GRID_NONUNIFORM),
        ("negative", Reason.DATA_INVALID),
        ("premise", Reason.METADATA_MISSING),
        ("support", Reason.UNKNOWN_BANDWIDTH),
    ],
)
def test_grid_and_evidence_refusals(full_delivery, case, reason):
    """Fails if corrupted coordinates, samples or unknown premises admit."""
    data = dict(full_delivery)
    if case == "duplicate":
        data["indices_xy"] = data["indices_xy"].at[0].set(data["indices_xy"][1])
    elif case == "missing":
        data["intensity_nodes"] = data["intensity_nodes"][:, 1:]
        data["indices_xy"] = data["indices_xy"][1:]
        data["offsets_xy_lod"] = data["offsets_xy_lod"][1:]
    elif case == "offset":
        data["offsets_xy_lod"] = data["offsets_xy_lod"].at[0, 0].add(1e-10)
    elif case == "negative":
        data["intensity_nodes"] = data["intensity_nodes"].at[0, 0, 0, 0].set(-1)
    elif case == "premise":
        data = changed(
            data,
            "products",
            lambda p: p.premises.evidence,
            jnp.zeros(6, dtype=jnp.int32),
        )
    else:
        data = changed(
            data,
            "products",
            lambda p: p.support_evidence,
            jnp.zeros(1, dtype=jnp.int32),
        )
    result = sc.admit_grid(**data)
    assert not result.admitted and result.reason_bits & reason
    with pytest.raises(sc.AdmissionRefused) as error:
        result.require()
    assert error.value.receipt is result.receipt


@pytest.mark.parametrize(
    "case,reason",
    [
        ("normalization", Reason.NORMALIZATION_UNSUPPORTED),
        ("conflict", Reason.METADATA_CONFLICT),
        ("frame", Reason.FRAME_UNSUPPORTED),
        ("halo", Reason.INCOMPLETE_STENCIL),
        ("memory", Reason.MEMORY_LIMIT),
        ("layout", Reason.LAYOUT_UNSUPPORTED),
    ],
)
def test_metadata_geometry_resources(full_delivery, case, reason):
    """Fails if unsupported units, frames, layouts or allocations are repaired."""
    data = dict(full_delivery)
    if case == "normalization":
        values = data["products"].premises.values.at[4].set(False)
        data = changed(data, "products", lambda p: p.premises.values, values)
    elif case == "conflict":
        data = changed(
            data, "products", lambda p: p.sampling_min_nm, jnp.array([600.0])
        )
    elif case == "frame":
        products = data["products"]
        data["products"] = replace(
            products, pixels=replace(products.pixels, basis=((0.25, 0.1), (0.0, 0.25)))
        )
    elif case == "halo":
        data["domain"] = sc.Domain(
            "box", "closed", jnp.array([[4.0, 4.0], [5.0, 5.0]]), jnp.zeros(2)
        )
    elif case == "memory":
        data["memory"] = sc.MemoryBudget(1, 1, 1, 1)
    else:
        r = data["reflection"]
        data["reflection"] = Reflection(
            "radial",
            r.axes,
            r.declaration_id,
            r.argument_id,
            r.evidence,
            r.check_report_id,
            r.check_passed,
            r.tolerance,
        )
    result = sc.admit_grid(**data)
    assert not result.admitted and result.reason_bits & reason


@pytest.mark.parametrize(
    "case,reason",
    [
        ("missing", Reason.SYMMETRY_CHECK_MISSING),
        ("invalid", Reason.SYMMETRY_REFERENCE_INVALID),
        ("failed", Reason.SYMMETRY_CHECK_FAILED),
        ("declaration", Reason.SYMMETRY_UNSUPPORTED),
        ("center", Reason.FRAME_UNSUPPORTED),
        ("origin", Reason.GRID_NONUNIFORM),
    ],
)
def test_quadrant_refusals(quadrant_delivery, case, reason):
    """Fails if caller flags bypass symmetry checks or midpoint/zero-anchor rules."""
    data = dict(quadrant_delivery)
    data = changed(data, "reflection", lambda r: r.check_passed, jnp.array([True]))
    if case == "missing":
        data["mirror_checks"] = None
    elif case == "invalid":
        data = changed(
            data,
            "mirror_checks",
            lambda c: c.intensity,
            data["mirror_checks"].intensity.at[0, 0, 0].set(jnp.nan),
        )
    elif case == "failed":
        data = changed(
            data,
            "mirror_checks",
            lambda c: c.intensity,
            data["mirror_checks"].intensity.at[0, 0, 0].add(0.1),
        )
    elif case == "declaration":
        data = changed(data, "reflection", lambda r: r.axes, jnp.array([True, False]))
    elif case == "center":
        products = data["products"]
        data["products"] = replace(
            products, pixels=replace(products.pixels, center_x=3.0, center_y=4.0)
        )
    else:
        grid = data["declared_lattice"]
        data["declared_lattice"] = source_grid(
            [0.25, 0.0], spacing_xy(grid), grid.shape, grid.frame
        )
    result = sc.admit_grid(**data)
    assert not result.admitted and result.reason_bits & reason


@pytest.mark.parametrize(
    "pitch,campaign,margin,nyquist",
    [
        (0.38, math.nan, False, False),
        (math.nextafter(0.38, math.inf), math.nan, True, False),
        (0.5, math.nan, True, True),
        (0.342, 450.0, True, False),
        (0.34199999999999997, 450.0, False, False),
    ],
)
def test_literal_sampling_threshold(full_delivery, pitch, campaign, margin, nyquist):
    """Fails on tolerance-relaxed margins or reassociated wavelength arithmetic."""
    grid = full_delivery["declared_lattice"]
    data = dict(
        full_delivery,
        declared_lattice=source_grid(
            origin_xy(grid), [pitch, pitch], grid.shape, grid.frame
        ),
    )
    data = changed(
        data, "products", lambda p: p.campaign_sampling_min_nm, jnp.array(campaign)
    )
    data["offsets_xy_lod"] = origin_xy(data["declared_lattice"]) + data[
        "indices_xy"
    ] * spacing_xy(data["declared_lattice"])
    # Keep permission in the interior after changing pitch; no endpoint fitting.
    midpoint = origin_xy(data["declared_lattice"]) + 20 * pitch
    data["domain"] = sc.Domain(
        "box", "closed", jnp.stack((midpoint - 0.1, midpoint + 0.1)), jnp.zeros(2)
    )
    result = sc.admit_grid(**data)
    assert bool(result.reason_bits & Reason.KERNEL_MARGIN_FAIL) == margin
    assert bool(result.reason_bits & Reason.NYQUIST_FAIL) == nyquist
    assert result.admitted == (not margin and not nyquist)


def test_shuffled_coordinates_and_float32_offsets(full_delivery):
    """Fails on trusted row order or a float64 ULP allowance for float32 offsets."""
    base = sc.admit_grid(**full_delivery)
    data = dict(full_delivery)
    order = jax.random.permutation(jax.random.key(0), data["indices_xy"].shape[0])
    data["intensity_nodes"] = data["intensity_nodes"][:, order].astype(jnp.float32)
    data["indices_xy"] = data["indices_xy"][order]
    data["offsets_xy_lod"] = data["offsets_xy_lod"][order].astype(jnp.float32)
    result = sc.admit_grid(**data)
    np.testing.assert_array_equal(
        result.require().intensity, base.require().intensity.astype(jnp.float32)
    )
    assert result.receipt.offset_dtype == "float32"
    assert result.receipt.floor_rule == "exact_quotient_floor_v1"
    assert "exact_quotient_floor_v1" in result.receipt.profile_id


@pytest.mark.float64_only(reason="Float64 reference roundoff for empirical residuals")
def test_checkset_is_empirical(full_delivery):
    """Fails if independent residuals become a bound or lose their scope."""
    positions = jnp.array([[0.13, -0.21], [-0.21, 0.13]])
    truth = jax.vmap(analytic)(positions)
    peaks = jnp.array([3.0, 4.0])
    checks = CheckSet(
        positions,
        truth,
        jnp.zeros(2, dtype=jnp.int32),
        peaks,
        "independent",
        "closed_form",
        "synthetic",
    )
    data = {**full_delivery, "checks": checks}
    admission = sc.admit_grid(**data)
    reader = Interpolator(
        admission.require(),
        data["domain"],
        sc.KernelProfile(),
        sc.ReadPolicy(),
        sc.Tiling(),
        data["memory"],
    )
    result = eqx.filter_jit(read_batch)(reader, positions, checks.product_index)
    residual = jnp.max(jnp.abs(result.intensity - truth), axis=(1, 2))
    np.testing.assert_allclose(
        admission.receipt.empirical_absolute, residual, atol=3e-15, rtol=0
    )
    np.testing.assert_allclose(
        admission.receipt.empirical_local_peak, residual / peaks, atol=1e-15, rtol=0
    )
    np.testing.assert_array_equal(
        result.empirical_max_abs, jnp.full(2, admission.receipt.empirical_max_abs)
    )
    assert bool(jnp.all((result.reason_bits & Reason.EMPIRICAL_ONLY) != 0))
    assert bool(jnp.all(jnp.isinf(result.abs_error_bound)))


@pytest.mark.float64_only(
    reason="Compares a float64 reader with and without float64 available"
)
def test_precision_and_allocation_limits(full_delivery, full_reader):
    """Fails if x64-off silently truncates or batches allocate past byte ceilings."""
    with jax.enable_x64(False):
        admission = sc.admit_grid(**full_delivery)
        assert admission.reason_bits == Reason.PRECISION_UNAVAILABLE
        with pytest.raises(sc.PrecisionUnavailable):
            admission.require()
        with pytest.raises(sc.PrecisionUnavailable):
            full_reader.read(jnp.array([0.0, 0.0]), jnp.int32(0))
    memory = sc.MemoryBudget(100_000_000, 1000, 10_000_000, 400_000_000)
    reader = eqx.tree_at(lambda r: r.memory, full_reader, memory)
    with pytest.raises(sc.MemoryLimit):
        read_batch(reader, jnp.zeros((10, 2)), jnp.zeros(10, dtype=jnp.int32))
    tiny = {**full_delivery, "memory": sc.MemoryBudget(1, 1, 1, 1)}
    with pytest.raises(sc.MemoryLimit):
        sc.admit_grid(**tiny).require()


def test_axis_invariance_and_odd_detector(quadrant_delivery):
    """Fails if stored axes escape checks; odd frames use the same midpoint rule."""
    data = dict(quadrant_delivery)
    # The independent mirrors use (1,1); alter a different axis node only.
    data["intensity_nodes"] = data["intensity_nodes"].at[0, 0, 0, 0].add(0.01)
    result = sc.admit_grid(**data)
    assert result.reason_bits & Reason.SYMMETRY_CHECK_FAILED
    odd = delivery(quadrant=True, shape=(7, 5))
    assert make_reader(odd).grid.reflection.check_passed[0]
