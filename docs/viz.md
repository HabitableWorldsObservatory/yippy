---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Plot performance curves

Install `yippy[viz]` to draw curves with eyepiece. All compute methods return
data and take no `plot` argument. Plotting lives in `yippy.viz`, with no
top-level plotting exports. Importing either module leaves the style unchanged.

The plotting functions accept precomputed arrays; `plot_performance_curve`
also accepts a `PerformanceCurves` record and a `metric` name. They return
eyepiece `PlotResult` objects and draw into a single caller-supplied `ax`, or
create an axes when it is omitted. They never show or save figures themselves.

This example uses synthetic arrays with a known constant stellar density.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from hwoutils.grids import Frame

from optixstuff import DEFAULT_APERTURE_RULE, PerformanceCurves
from yippy import viz

hwostyle.use("light")
separations = np.linspace(0, 8, 40)
throughput = 0.3 * (1 - np.exp(-separations**2 / 4)) + 1e-6
area = np.full_like(separations, np.pi * DEFAULT_APERTURE_RULE.radius_lod**2)
density = np.full_like(separations, 1e-10)
curves = PerformanceCurves(
    separation_lod=separations,
    frame=Frame(500, 6, "circumscribed"),
    rule=DEFAULT_APERTURE_RULE,
    throughput=throughput,
    core_area_lod2=area,
    leakage_fraction=density * area,
    leakage_density_lod2=density,
    raw_contrast=density * area / throughput,
    iwa_lod=2 * np.sqrt(np.log(2)),
    data_extent_lod=8,
)
result = viz.plot_performance_curve(curves, metric="raw_contrast")
result.ax.set_title("Synthetic raw contrast")
plt.show()
```

For an occulter transmission curve, pass its arrays and name the metric.

```{code-cell} python
transmission = 1 - np.exp(-separations**2 / 4)
result = viz.plot_performance_curve(
    separations, transmission, metric="occulter_transmission"
)
plt.show()
```

`plot_core_mean_intensity` consumes a separation array and diameter-to-profile
mapping. Build each profile with
`stellar_intensity_curves(coro, diameter_lod).evaluate("core_mean_intensity", separations)`.
Values are per pixel, not per unit angular area. Distinct colors and markers
distinguish diameters.

```{code-cell} python
profiles = {
    0.0: 1e-10 * (1 + np.exp(-separations)),
    0.1: 2e-10 * (1 + np.exp(-separations)),
}
result = viz.plot_core_mean_intensity(separations, profiles)
result.ax.legend(frameon=False)
plt.show()
```

Supported record metrics are `throughput`, `core_area_lod2`,
`leakage_fraction`, `leakage_density_lod2`, and `raw_contrast`.
Stellar leakage and raw contrast use logarithmic y axes. Zero and negative
samples remain in the line data; Matplotlib's log-axis policy controls their
display. Use `line_kw` for line styling and the returned axes for titles and
annotations. Pass colors and markers from a shared `eyepiece.style.SourceStyles`
through `line_kw` when comparing curves across figures.

## Plotting a loaded model

Prepare the desired rules before selecting a record. No mutable calculation
method is needed after loading:

```python notest
from optixstuff import Circular
from yippy import Coronagraph, viz

rule = Circular(0.7)
coro = Coronagraph("path/to/yip", rules=(rule,))
result = viz.plot_performance_curve(coro.curves(rule), metric="throughput")
```
