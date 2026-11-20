"""Sampling-admitted intensity grids and windowed-sinc reads.

The package exports what a caller of ``SincOptions`` and a scene caller need.
Admission rules, planners, evidence types and the reader internals stay
importable from their own modules (``yippy.sinc.admission``, ``.planner``,
``.evidence``, ``.reader``, ``.kernel``).
"""

from yippy.sinc.domain import Domain
from yippy.sinc.evidence import (
    AdmissionRefused,
    MemoryLimit,
    PrecisionUnavailable,
    ReadRefused,
)
from yippy.sinc.grid import Premises, Products, SampledGrid, admit_grid
from yippy.sinc.kernel import KernelProfile
from yippy.sinc.reader import MemoryBudget, ReadPolicy, Tiling, make_reader
from yippy.sinc.scene import SceneRenderer

__all__ = [
    "AdmissionRefused",
    "Domain",
    "KernelProfile",
    "MemoryBudget",
    "MemoryLimit",
    "PrecisionUnavailable",
    "Premises",
    "Products",
    "ReadPolicy",
    "ReadRefused",
    "SampledGrid",
    "SceneRenderer",
    "Tiling",
    "admit_grid",
    "make_reader",
]
