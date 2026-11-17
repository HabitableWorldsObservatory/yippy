# Contrast Floors

Performance records store the measured raw contrast without an engineering
floor. Apply an engineering floor explicitly during post-processing of
contract quantities:

```python notest
import numpy as np
from optixstuff import YippyCoronagraph
from yippy import Circular

optics = YippyCoronagraph("path/to/yip")
rule = Circular(0.7)
curves = optics.curves(rule)
separations = np.array([3.0, 5.0, 10.0, 20.0])
contrast = np.maximum(np.abs(optics.raw_contrast(separations, rule=rule)), 4e-10)
noise = contrast / 30.0
```

Smaller contrast means better suppression, so the engineering floor is a
lower bound: `max(abs(raw_contrast), contrast_floor) / ppf` gives the
EXOSIMS noise convention. It changes neither the stored curve nor the PSFs,
aperture throughput, aperture area, stellar leakage, or sky transmission.
AYO density noise is `optics.leakage_density_lod2(separations) / ppf`;
per-stellar-pixel ring intensity is read with
`stellar_intensity_curves(backend).evaluate("core_mean_intensity", separations)`.
Multiply density by the stellar pixel area only when a per-pixel consumer
requires it, accounting for the two products' Frames when they differ.

See {doc}`/examples/06_Noise_Floor_Conventions` for a per-pixel comparison.

The residual fraction belongs to `jaxedith.PostProcessing.speckle_residual_fraction`
(`1 / ppf` for this convention), rather than to the coronagraph.
