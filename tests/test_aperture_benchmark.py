"""The maintained benchmark runs without a Git checkout or test imports."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_benchmark_airy_reference_without_git(tmp_path):
    """Analytic evidence regenerates from the installed code with no Git tool."""
    script = Path("tools/benchmark_apertures.py").resolve()
    env = dict(os.environ, PATH="")
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", str(Path("src").resolve()))
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path), "--airy-only"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    with (tmp_path / "airy.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20
    errors = [
        abs(float(row["measured_10x"]) / float(row["analytic"]) - 1) for row in rows
    ]
    assert np.max(errors) < 1e-3
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["references"]["airy"]
    assert provenance["oversample"] == [10, 40]
    assert not list(Path("results/aperture-convergence").glob("*.csv"))
