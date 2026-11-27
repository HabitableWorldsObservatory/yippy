# Contrast Floors

Performance records store the measured raw contrast without an engineering
floor. Apply an engineering floor explicitly during post-processing of
contract quantities:

```python notest
import numpy as np
from optixstuff import Circular
from yippy import open_yip

rule = Circular(0.7)
coro = open_yip("path/to/yip").prepare(rules=(rule,))
curves = coro.curves(rule)
separations = np.array([3.0, 5.0, 10.0, 20.0])
contrast = np.maximum(np.abs(curves.evaluate("raw_contrast", separations, 0.0)), 4e-10)
noise = contrast / 30.0
```

Smaller contrast means better suppression, so the engineering floor is a
lower bound: `max(abs(raw_contrast), contrast_floor) / ppf` gives the
EXOSIMS noise convention. It changes neither the stored curve nor the PSFs,
aperture throughput, aperture area, stellar leakage, or sky transmission.
The AYO CSV noise-floor column gives the exported point-source contrast
divided by `ppf`. The corresponding residual density is
`curves.evaluate("leakage_density_lod2", separations, 0.0) / ppf`;
per-stellar-pixel ring intensity is read with
`yippy.performance.stellar_intensity_curves` as
`stellar_intensity_curves(coro).evaluate(separations)`.
Multiply density by the stellar pixel area only when a per-pixel consumer
requires it, accounting for the two products' Frames when they differ.

See {doc}`/examples/06_Noise_Floor_Conventions` for a per-pixel comparison.

The residual fraction belongs to `optixstuff.PostProcessing.speckle_residual_fraction`
(`1 / ppf` for this convention), rather than to the coronagraph.
