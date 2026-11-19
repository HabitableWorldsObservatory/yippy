"""Optional plotting imports remain lazy and leave style state alone."""

import subprocess
import sys


def test_base_install_and_actionable_error():
    code = """
import sys
sys.modules['eyepiece'] = None
import yippy.sinc
import yippy.sinc.viz as viz
assert 'matplotlib' not in sys.modules
assert 'plot_kernel' in dir(viz)
try:
    viz.plot_kernel
except ImportError as error:
    assert "pip install 'yippy[viz]'" in str(error)
else:
    raise AssertionError('missing optional dependency was accepted')
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr


def test_exports_have_no_style_side_effects():
    code = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yippy.sinc.viz as viz
before = dict(matplotlib.rcParams)
for name in viz.__all__:
    assert name in dir(viz)
    assert callable(getattr(viz, name))
assert before == dict(matplotlib.rcParams)
assert not plt.get_fignums()
try:
    viz.missing
except AttributeError:
    pass
else:
    raise AssertionError('unknown name was accepted')
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
