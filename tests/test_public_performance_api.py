"""Every function advertised by the performance documentation is exported."""

import ast
import json
from pathlib import Path

import pytest
from optixstuff.apertures import DEFAULT_APERTURE_RULE, Circular

import yippy
from yippy import performance


def test_documented_performance_names_are_exported():
    """Notebook imports and the measurement kernels are supported package API."""
    names = {"stellar_intensity_curves", "core_separations"}
    for notebook in Path("docs/examples").glob("*.ipynb"):
        for cell in json.loads(notebook.read_text())["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if source.startswith("%"):
                continue
            for node in ast.walk(ast.parse(source)):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "yippy.performance"
                ):
                    names.update(
                        alias.name
                        for alias in node.names
                        if not alias.name.startswith("_")
                    )
    for name in sorted(names):
        assert callable(getattr(performance, name))
    for name in (
        "EqxCoronagraph",
        "Circular",
        "HalfMaximum",
        "PerformanceCurves",
        "performance_curves",
    ):
        assert not hasattr(yippy, name)

    for name in (
        "compute_core_mean_intensity_curve",
        "core_mean_intensity_profile",
        "compute_occ_trans_curve",
    ):
        assert not hasattr(performance, name)


@pytest.mark.parametrize(
    ("rule", "leakage_rule"),
    [(Circular(0.8), "aperture_mean"), (DEFAULT_APERTURE_RULE, "azimuthal_ring_mean")],
)
def test_with_rules_prepares_the_requested_rule(
    cached_diameter_coro, rule, leakage_rule
):
    """Preparing a missing record returns a new model and retains the original."""
    coro = cached_diameter_coro
    record = coro.curves()
    with pytest.raises(ValueError, match="No prepared record"):
        coro.curves(rule, leakage_rule=leakage_rule)

    model = coro.with_rules((rule,), leakage_rule=leakage_rule)
    assert model is not coro
    prepared_record = model.curves(rule, leakage_rule=leakage_rule)
    assert prepared_record.rule == rule
    assert prepared_record.leakage_rule == leakage_rule
    assert coro.curves() is record
