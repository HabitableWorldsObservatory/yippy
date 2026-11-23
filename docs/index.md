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
examples/SkyTrans.ipynb
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
:caption: Sinc reads
sinc/index.md
```

```{toctree}
:maxdepth: 2
:caption: Reference
stellar_map.md
viz.md
contrast_floor.md
azimuthal_averaging.md
circle_reader_accuracy.md
benchmarking.md
sinc-delivery.md
```

```{toctree}
:maxdepth: 1
:caption: Development
develop/_changelog.md
```

## Performance API

`yippy.open_yip(path).prepare(rules=(...))` directly implements
`optixstuff.AbstractTableCoronagraph`. Supply aperture rules at construction,
select records with `coro.curves(rule)`, and query images through
`source_psfs(None, x_lod, y_lod, grid=None).image` or
`stellar_map(None, diameter_lod, grid=None)`. Native images carry their product grid and sampling measure. Point samples
carry density per square grid unit. Call `image.integrated()` for midpoint
photon fractions per pixel. Explicit target grids request pixel integration
(see [the stellar intensity map](stellar_map.md) for the read of the star).
Source reads include one validity flag per position.

Aperture rules and generic photometry belong to optixstuff:
`optixstuff.Circular`, `HalfMaximum`, `PerformanceCurves`, and
`optixstuff.performance.performance_curves`. YIP-specific radial diagnostics
remain in `yippy.performance`. Names beginning with `_` are implementation
details.

For photometry on an arbitrary image at a known centroid, pass an explicit
`fixed_center=(row, column)` in original pixel coordinates:

This example needs a user-supplied image (`image`).

```python notest
from optixstuff.performance import circular_aperture

flux, area, _ = circular_aperture(
    image, radius_pix=2.8, pixel_scale_lod=0.25,
    fixed_center=(32.2, 31.7),
)
```

This center takes precedence over `nominal_center` and performs no peak search.
The window must contain the aperture, and image edges clip both flux and area.
