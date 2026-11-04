# Welcome to yippy Documentation

`yippy` is a package designed to read a "Yield Input Package" (YIP) in Python and generate the off-axis PSFs, stellar intensity maps, and sky transmission maps.

## Quick Links

```{toctree}
:maxdepth: 2
:caption: Getting Started
examples/Basics.ipynb
datasets.md
```

```{toctree}
:maxdepth: 2
:caption: YIP Internals
examples/RadiallySymmetric.ipynb
examples/QuarterSymmetric.ipynb
examples/StellarIntens.ipynb
examples/SkyTrans.ipynb
examples/FourierInterpolation.ipynb
```

```{toctree}
:maxdepth: 2
:caption: Performance Metrics
examples/00_Performance_Metrics_Overview.ipynb
examples/01_Core_Throughput.ipynb
examples/02_Stellar_Leakage_and_Contrast.ipynb
examples/03_Spatial_Metrics_and_Backgrounds.ipynb
examples/04_Noise_Floors_and_Integration_Time.ipynb
examples/05_Aperture_Methods_Comparison.ipynb
examples/06_Noise_Floor_Conventions.ipynb
```

```{toctree}
:maxdepth: 2
:caption: Reference
contrast_floor.md
azimuthal_averaging.md
circle_reader_accuracy.md
benchmarking.md
```

```{toctree}
:maxdepth: 1
:caption: Development
develop/_changelog.md
```

## Performance API

The performance functions imported in these examples are supported public API
and are also exported from `yippy`. They include `compute_truncation_curves`,
`truncation_core`, and `circular_aperture`. Names beginning with `_` are
implementation details, even when an example uses them to display intermediate
samples.

For photometry on an arbitrary image at a known centroid, pass an explicit
`fixed_center=(row, column)` in original pixel coordinates:

This example needs a user-supplied image (`image`).

```python notest
from yippy import circular_aperture

flux, area, _ = circular_aperture(
    image, radius_pix=2.8, pixel_scale_lod=0.25,
    fixed_center=(32.2, 31.7),
)
```

This center takes precedence over `nominal_center` and performs no peak search.
The window must contain the aperture, and image edges clip both flux and area.
