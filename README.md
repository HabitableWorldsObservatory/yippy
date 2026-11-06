<p align="center">
  <img width = 250 src="https://raw.githubusercontent.com/HabitableWorldsObservatory/yippy/main/docs/_static/logo.png" alt="yippy logo" />
  <br><br>
</p>

<p align="center">
  <a href="https://pypi.org/project/yippy/"><img src="https://img.shields.io/pypi/v/yippy.svg?style=flat-square&logo=pypi" alt="PyPI"/></a>
  <a href="https://yippy.readthedocs.io"><img src="https://readthedocs.org/projects/yippy/badge/?version=latest&style=flat-square" alt="Documentation Status"/></a>
  <a href="https://github.com/HabitableWorldsObservatory/yippy/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/HabitableWorldsObservatory/yippy/ci.yml?branch=main&style=flat-square&logo=github&label=tests" alt="Tests"/></a>
  <a href="https://github.com/HabitableWorldsObservatory/yippy/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"/></a>
  <a href="https://github.com/HabitableWorldsObservatory/yippy/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg?style=flat-square&logo=python" alt="Python 3.12+"/></a>
  <a href="https://doi.org/10.5281/zenodo.20172435"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20172435-blue?style=flat-square" alt="DOI"/></a>
</p>

---

# yippy

A Python and JAX library for loading coronagraph yield input packages (YIPs)
and computing coronagraph performance metrics. yippy provides Fourier-based
off-axis PSF interpolation, throughput/contrast/core-area curves, 2D
performance maps, and export to EXOSIMS and AYO formats.

## Installation

```bash
pip install yippy
```

## Quick Start

The primary workflow is to point yippy at a YIP directory you already have
on disk -- either one you built yourself or one you downloaded:

This example needs a user-supplied YIP directory.

```python notest
from yippy import Coronagraph

# Load a YIP from a folder on disk
coro = Coronagraph("path/to/eac1_aavc_2d")

# Off-axis PSF at a given (x, y) position
from lod_unit import lod

offaxis_psf = coro.offax(2 * lod, 5 * lod)

# Performance metrics at any separation
throughput = coro.throughput(5.0)  # scalar or array
contrast = coro.raw_contrast(5.0)
occ_trans = coro.occulter_transmission(5.0)

# 2D performance maps (pixel grids)
throughput_map = coro.throughput_map()
core_area_map = coro.core_area_map()
```

If you don't have a YIP on hand, see [Example data](#example-data-beta)
below for a convenience downloader.

## Two-Class Design

yippy provides two coronagraph classes for different use cases:

| | `Coronagraph` | `EqxCoronagraph` |
|---|---|---|
| **Purpose** | Full-featured analysis & export | JIT-compiled simulation |
| **Backend** | NumPy/SciPy + JAX | Pure JAX/Equinox |
| **JIT-compatible** | No | Yes (`eqx.filter_jit`) |
| **GPU/TPU support** | PSF generation only | Everything |
| **I/O & export** | EXOSIMS FITS, AYO CSV | None (simulation only) |
| **Performance curves** | Computed on init | Converted from `Coronagraph` |

### `Coronagraph` -- Analysis & Data Management

The primary class for loading YIPs, computing performance curves, and
exporting to external formats:

This example needs a user-supplied YIP directory.

```python notest
from yippy import Coronagraph

coro = Coronagraph("path/to/yip")

# 1D performance curves (scalar or array separations in lam/D)
coro.throughput(5.0)
coro.raw_contrast(5.0)
coro.occulter_transmission(5.0)
coro.core_area(5.0)
coro.core_mean_intensity(5.0)

# Noise floors
coro.noise_floor_exosims(5.0)  # |raw_contrast| / ppf
coro.noise_floor_ayo(5.0, ppf=30.0)  # core_mean_intensity / ppf

# 2D maps (full pixel grids)
coro.separation_map()
coro.throughput_map()
coro.core_area_map()
coro.core_mean_intensity_map()
coro.noise_floor_ayo_map(ppf=30.0)

# Export
coro.to_exosims()
coro.dump_ayo_csv("output.csv")
```

### `EqxCoronagraph` -- JIT-Compatible Simulation

A pure JAX/Equinox module for use inside `jax.jit`-compiled pipelines:

This example needs a user-supplied YIP directory.

```python notest
from yippy import EqxCoronagraph
import equinox as eqx

# Create from a YIP path directly
coro = EqxCoronagraph("path/to/yip")


# All methods are JIT-traceable
@eqx.filter_jit
def simulate(coro, x, y):
    psf = coro.create_psf(x, y)
    stellar = coro.stellar_intens(0.01)
    throughput = coro.throughput(5.0)
    return psf, stellar, throughput
```

## Performance Metrics

Individual metric functions are available in `yippy.performance` for
standalone analysis:

This example needs a configured coronagraph (`coro`).

```python notest
from yippy.performance import (
    compute_throughput_curve,
    compute_raw_contrast_curve,
    compute_core_area_curve,
    compute_occ_trans_curve,
    compute_core_mean_intensity_curve,
    compute_truncation_throughput_curve,  # PSF truncation-ratio aperture
    compute_truncation_core_area_curve,
)

# Compute individual curves
separations, throughputs = compute_throughput_curve(coro)
separations, contrasts = compute_raw_contrast_curve(coro)
```

### PSF Truncation Ratio

When `psf_trunc_ratio` is set (e.g. `Coronagraph(path, psf_trunc_ratio=0.3)`),
throughput and core area are computed using an adaptive aperture that includes
all oversampled pixels exceeding `ratio * peak`. This matches AYO's
`photap_frac` / `omega_lod` calculation and is recommended for ETC integration.
Core areas and aperture throughputs use the PSF's band-limited (trigonometric)
interpolant at 10 subpixels per pixel side. Circular apertures use exact
circle-square overlap fractions; truncation cores use fractional areas from
the local level-set tangent, with gradients from the same DFT basis. Flux
and area use the same weights. A circle fully inside the image has area
`pi * radius**2`. Gaussian core-area fits remain fits.

Peak searches stay within `PEAK_SEARCH_RADIUS_LOD = 1.0` lambda/D of the
nominal source position, including for raw contrast. Stellar leakage uses
that same planet-centered circle. Truncation measurements include only the
local core window for in-image sources, preventing distant leakage speckles
from setting the core below the IWA. Native offsets outside the image retain
the global in-image truncation-core convention.

By default the curves are tabulated only at the package's own non-negative
off-axis offsets and interpolated between them. Passing
`separation_step_lod=CORE_STEP_LOD` to `compute_truncation_curves` or
`compute_all_performance_curves` opts into a 0.25 lambda/D grid of synthesized
PSFs as well. A YIP alone cannot establish whether blended PSFs reproduce the
optics between shipped offsets.

Band-limited reconstruction requires sampling at 0.5 lambda/D per pixel or
finer. Coarser packages emit one `UserWarning` per `Coronagraph` and still
compute. Fractional boundaries remove the lattice-point area error;
intensity quadrature, level-set curvature, and finite-image boundaries still
require oversampling convergence checks.

## Example data

A small set of reference YIPs ships as GitHub release assets on this repo
and can be fetched on demand. This is useful for tutorials, CI, and "I just
want to try it" exploration:

```python
from yippy import Coronagraph, fetch_yip

yip_path = fetch_yip("eac1_aavc_2d")  # download (cached after first call)
coro = Coronagraph(yip_path)
```

The catalog is currently minimal. Long-term YIP hosting will be provided
by ExEP, and only a handful of reference YIPs from the Coronagraph Design
Survey are shipped here: `eac1_aavc_2d` and `eac1_optimal_order_6_1d`
(used by the yippy paper validation pipeline and as the pyEDITH default
coronagraph) and `usort_offaxis_ovc` (an off-axis vortex coronagraph
design study). For production work or YIPs outside this set, manage your
own YIP paths and pass them to `Coronagraph(path)` directly.

The `eac1_aavc_2d` design is Susan Redmond's EAC1 apodized vortex
coronagraph, described in D. P. Mawet et al., "Apodized vortex
coronagraph for the Habitable Worlds Observatory," Proc. SPIE **13627**
(2025), [doi:10.1117/12.3065779](https://doi.org/10.1117/12.3065779).
Cite the design alongside yippy when you publish results built on it.

See [`yippy.list_yips()`](https://yippy.readthedocs.io/en/latest/datasets.html)
for the available names and the
[datasets guide](https://yippy.readthedocs.io/en/latest/datasets.html)
for filtered queries and metadata inspection.

### Cache location (advanced)

By default, `fetch_yip` caches archives under
`pooch.os_cache("yippy")` -- the OS-conventional location provided by
[platformdirs](https://platformdirs.readthedocs.io) (e.g.
`~/Library/Caches/yippy` on macOS, `~/.cache/yippy` on Linux). To pin
the cache elsewhere:

```bash
# Persistent override -- pyEDITH and other consumers inherit this
export YIPPY_CACHE_DIR=~/Documents/YIPs
```

This example downloads a YIP into a user-configured shared cache directory.

```python notest
# Per-call override (wins over the env var)
yip_path = fetch_yip("eac1_aavc_2d", cache_path="/data/shared/yips")

# Introspect the resolved cache directory without triggering a fetch
from yippy import cache_dir

print(cache_dir())
```

## Precision

`yippy` follows JAX's global `jax_enable_x64` flag. The default is `float32`
(JAX's default and the memory-friendly choice -- the PSF datacube dominates
memory). For a uniform `float64` pipeline -- needed for high-fidelity work or
backends that require double precision -- enable x64 *before* building a
coronagraph:

This example needs a user-supplied YIP directory.

```python notest
import jax

jax.config.update("jax_enable_x64", True)  # or set JAX_ENABLE_X64=1 in the env

from yippy import Coronagraph, EqxCoronagraph

coro = EqxCoronagraph("/path/to/yip")  # now all arrays are float64
```

There is no per-object dtype option: precision is one global switch, so the
whole pipeline stays a single dtype (JAX does not mix `float32` and `float64`
cleanly). PSF datacube caches are dtype-keyed (`psf_datacube*_f32.npy` vs
`*_f64.npy`), so float32 and float64 runs never share a cache.

## Units

Yield input packages use $`\lambda / D`$ units so `yippy` treats them
as the default and uses the `lod_unit` package to define the `lod` unit. However,
it can use three different `astropy` units: pixels (as defined by the yield
input package), angular separation (angle units), or apparent separation
(length units). If no units are provided it assumes the input is in $`\lambda / D`$.

This example needs a configured coronagraph (`coro`).

```python notest
import astropy.units as u

# pixels
x_pos = 2 * u.pix
y_pos = 5 * u.pix
offaxis_psf = coro.offax(x_pos, y_pos)

# angular separation
telescope_diameter = 10 * u.m
wavelength = 500 * u.nm
offaxis_psf = coro.offax(x_pos, y_pos, lam=wavelength, D=telescope_diameter)

# apparent separation
star_dist = 10 * u.pc
offaxis_psf = coro.offax(
    x_pos, y_pos, lam=wavelength, D=telescope_diameter, dist=star_dist
)
```

## JAX

yippy uses [JAX](https://jax.readthedocs.io) for JIT compilation and
GPU/TPU-accelerated PSF generation. JAX defaults to 32-bit precision;
if you need 64-bit precision, configure it
**before** importing yippy or any other JAX-based library:

```python
# At the very top of your script
from hwoutils import enable_x64, set_platform

enable_x64()  # switch to float64
set_platform("cpu")  # or "gpu", "gpu,cpu" for fallback

# Now it's safe to import yippy
from yippy import Coronagraph
```

Or via environment variables (safest):

```bash
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu python my_script.py
```

See the [JAX Configuration Guide](https://github.com/CoreySpohn/hwoutils/blob/main/docs/jax_configuration.md)
in hwoutils for details and common gotchas.

### Constructor options

- `use_jax`: Use JAX for PSF computation. Default is `True`.
- `x_symmetric`: Off-axis PSFs are symmetric about the x-axis. Default is `True`.
- `y_symmetric`: Off-axis PSFs are symmetric about the y-axis. Default is `True`.

### Parallel processing of off-axis PSFs

The base call of `coronagraph.offax(x,y)` is the most user-friendly, but is not
the most efficient. When generating many PSFs it is recommended to convert all
required (x,y) positions into arrays of floats (in $`\lambda / D`$) and use the
`coronagraph.offax.create_psfs_parallel(x_arr, y_arr)` function. This function
uses JAX's `shard_map` to distribute the computation across multiple CPU devices.

To use multiple CPU devices, call `hwoutils.set_host_device_count(N)` or set
`XLA_FLAGS=--xla_force_host_platform_device_count=N` **before** importing JAX.
On GPU/TPU backends, yippy automatically uses `vmap`+`jit` instead of `shard_map`.

### Optional sampled-intensity reader

Install `yippy[sinc]` to read directories containing a `sinc-delivery.json`
manifest with schema `sinc-delivery-0.1`. Enable JAX x64 before construction.
The `sinchole` dependency must be available separately until its first release.

```python notest
import jax
from yippy import Coronagraph, EqxCoronagraph, SincOptions

jax.config.update("jax_enable_x64", True)
coro = Coronagraph(
    "delivery", interpolation="sinc", sinc_options=SincOptions(product_id="mono-000")
)
image = coro.create_psf(4.0, 5.5)
result = coro.create_psf_with_info(4.0, 5.5)
view = EqxCoronagraph(yippy_coro=coro)
```

`blend` remains the default and does not import sinchole. Unmarked legacy
packages retain their numerical and cache behavior. `auto` records its
construction decision in the frozen `interpolation_info` record. It can use
blend for supported sampling refusals on full signed grids. Quadrant deliveries
require validated reflection evidence and midpoint detector geometry; they have
no blend fallback. Explicit `blend` rejects these deliveries.

Images contain incident photon fractions per pixel. Integral samples receive
only their declared photon conversion; mean densities additionally receive the
pixel area. Sinc output retains signed excursions. With-info reads expose status,
reason bits, converted error diagnostics and a `selected_backend` value of 0
(blend) or 1 (sinc). Sampling admission permits unknown error; bounded policies
refuse. Eager strict reads raise on refusal, while compiled reads return NaNs.

The adapter selects one monochromatic product and reads stellar and sky data
from the manifest's separate legacy package. The instance keeps the stellar
`header`, plus distinct `offaxis_header` and `sky_header` records. In the Equinox
view, `pixel_scale_lod` is the off-axis pitch; `stellar_pixel_scale_lod` and
`sky_pixel_scale_lod` describe the other maps. Its `source_psfs`, `stellar_map`
and `background_transmission` methods resample using their respective pitches.
Their wavelength argument preserves fixed-product table compatibility and does
not select another product. Target pitches and coordinates use the respective
product's reference frame. Wrappers with a single shared pitch must delegate
these methods explicitly to preserve distinct stellar and sky frames.

`SincOptions` accepts sinchole domain, memory and policy objects. Domain overrides
can only restrict the declared served geometry. Policy error budgets use photon
fractions per pixel. The default memory budget is computed from the actual
native cube size using sinchole's admission and 1024-output estimates, one-query
chunks and 256-pixel tiles. It excludes compiler workspace and caller-owned
arrays; larger batches need an explicit budget. Cube caches preserve per-query
status and backend selections in an integrity-checked sidecar.
