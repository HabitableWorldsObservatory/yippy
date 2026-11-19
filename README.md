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

A Python and JAX library for loading coronagraph yield input packages (YIPs),
interpolating PSFs, and exporting performance to EXOSIMS and AYO formats.
`yippy.Coronagraph` is an immutable Equinox module implementing
`optixstuff.AbstractTableCoronagraph` directly. Aperture rules and generic
performance records belong to optixstuff.

## Installation

```bash
pip install yippy
```

## Quick start

This example needs a user-supplied YIP directory. Source offsets, separations,
and stellar diameters are dimensionless numeric values in the native product's
lambda/D frame. Convert unit-bearing values before querying the model.

```python notest
from optixstuff import Circular, HalfMaximum
from yippy import open_yip

circle, core = Circular(0.7), HalfMaximum(0.3)
package = open_yip("path/to/yip")
native_psf = package.offaxis_psfs[0]
native_xy_lod = package.offsets_xy_lod[0]
coro = package.prepare(rules=(circle, core))

# Model PSFs evaluated at the requested positions, on the native pixel grid.
planet, valid = coro.source_psfs(None, [2.0], [5.0], grid=None)
star = coro.stellar_map(None, 0.01, grid=None)

# Explicit target grids request pixel integrals, including at native pitch.
rendered = coro.off_axis_psf(None, 5.0, coro.grid)
transmission = coro.uniform_sky_map(None, grid=coro.grid)

record = coro.curves(circle)
throughput = coro.throughput(5.0, rule=circle)
contrast = coro.raw_contrast(5.0, rule=circle)
area = record.evaluate("core_area_lod2", 5.0)
throughput_map = coro.throughput_map()
core_area_map = coro.core_area_map()
```

`package.prepare(...)` and `yippy.Coronagraph(package, ...)` build the same
model. `open_yip` reads the FITS headers and hashes the source files. It decodes the
image arrays on first access. The read-only arrays preserve sample order,
precision and zeros. `package.header.offaxis_psfs.pixel_grid()`,
`package.header.stellar_intensity.pixel_grid()` and
`package.header.sky_transmission.pixel_grid()` describe their individual grids.
Use `package.fits_header("offaxis_psfs")` for the original cards and comments.

`package.source.path` and `package.source.digest` identify the input.
Preparation reuses the package's open files. A metadata change on the same
file triggers a content check. Replacing a file requires a fresh `open_yip`.
`yippy.write_yip(destination, package)` copies the original products. It also
accepts measured arrays and their grids or headers from a producer.

Supply every required rule in `rules=` at construction. `curves(rule)` selects
a prepared record and raises for an unprepared rule. Reading a curve never
mutates the model. `coro.with_rules((rule, ...))` returns the same coronagraph
with a different set of prepared rules, without loading new tables. `leakage_rule="aperture_mean"` is the default;
`"azimuthal_ring_mean"` selects ring-mean leakage density at construction.
Pass the same `leakage_rule` to `curves` and scalar queries for that record.
`bandpass` is a `hwoutils.spectral.Band`, or `None` when the YIP has no edges.
`primary` reads both diameter conventions and collecting area from the YIP.

## Images, grids, and JAX

Images carry array values in `.data` and geometry in `.grid`.
`source_psfs` returns an `Image` with leading axis `"source"` and a boolean
validity vector. Stellar and sky maps are two-dimensional for scalar diameters.
Source and stellar
images are per-pixel fractions of unocculted stellar flux. Sky transmission is
dimensionless and uses sampled-value semantics. The off-axis, stellar, and sky
products carry separate grids, available from `coro.at_wavelength(wavelength_nm)`.
Model preparation requires matching off-axis and stellar grids for a five-file native YIP. A versioned sinc delivery can specify
its own off-axis grid. The sky map retains its own grid in both layouts.

Use `grid=None` to evaluate the prepared model on its native pixel grid.
The shipped source positions and images are available on the package. An explicit target
`PixelGrid` integrates the declared reconstruction onto its pixels. Native
coordinates remain in each product's lambda/D frame; convert a grid to sky
coordinates with `grid.to_arcsec(wavelength_nm)` when needed.

This example needs a configured `coro`.

```python notest
import equinox as eqx
import jax.numpy as jnp


@eqx.filter_jit
def simulate(coro, x_lod, y_lod, diameter_lod):
    planet = coro.source_psfs(None, x_lod, y_lod, grid=coro.grid)
    stellar = coro.stellar_map(None, diameter_lod, grid=coro.grid)
    return planet, stellar


planets, star = simulate(coro, jnp.array([3.0]), jnp.array([2.0]), 0.01)
```

For an extended scene, prepare its operator outside JIT. Unsupported pixels
may carry zero flux. An illuminated unsupported pixel sets the result validity
to false.

```python notest
coro = coro.with_scene_operator()
image, valid = coro.render_scene(scene_image, None, grid=detector_grid)
```

## Performance records

`coro.curves(rule)` returns an `optixstuff.PerformanceCurves` containing
`separation_lod`, `throughput`, `core_area_lod2`, `leakage_fraction`,
`leakage_density_lod2`, `raw_contrast`, `iwa_lod`, and `data_extent_lod`.
The corresponding variance moments are `throughput_variance`,
`core_area_variance_lod2`, `leakage_fraction_variance`, and
`leakage_density_variance_lod2`.
Use `record.evaluate(field, separation_lod)` to interpolate. Throughput and
area use linear-value PCHIP; leakage and contrast use log-value PCHIP. The
fields interpolate independently. Their contrast identity holds at measured
knots: contrast = leakage_density * core_area / throughput.

`Circular(radius_lod, centre="source")` fixes the aperture at the supplied
source position. `centre="peak"` follows the local PSF peak. `HalfMaximum(f)`
selects the connected core above fraction `f` of its peak. Both integrate
planet and stellar intensity over the same region and have identical mean
and variance moments. For inverse-variance photometry, pass `Matched()` to
`optixstuff.prepare_aperture` and supply the scene rates at the exposure query.

Default records measure the shipped non-negative off-axis samples. For
standalone aperture measurements on another separation axis, use
`optixstuff.performance.performance_curves`. Its source callback returns a
native two-dimensional `Image`; select the source axis from `source_psfs(...).image`
with `dataclasses.replace`, retaining the native grid and reconstruction.
The prepared `coro.curves(rule)` records require no callback.

Finite-star leakage queries accept `stellar_diam_lod=` and apply prepared
aperture operators to the interpolated stellar map. Throughput and area accept
the same keyword and are independent of stellar diameter. `time_d` is days
from the scene epoch and is unused by these fixed tables.

The separate stellar ring profile remains available from
`yippy.performance.stellar_intensity_curves(coro, diameter_lod)`. Its
`stellar_ring_mean_per_pixel` quantity gives per-stellar-pixel flux fractions
at `profile.evaluate(separation_lod)`. Divide by the
stellar grid pixel area for density per square lambda/D.

## Post-processing and export

Records store unfloored optical quantities. Apply an engineering contrast
floor and a post-processing factor explicitly in the consumer:

```python notest
import numpy as np

ppf = 30.0
contrast_noise = np.maximum(np.abs(coro.raw_contrast(5.0)), 4e-10) / ppf
density_noise = coro.leakage_density_lod2(5.0) / ppf
coro.to_exosims()
coro.dump_ayo_csv("output.csv")
```

`jaxedith.PostProcessing.speckle_residual_fraction` owns the residual fraction
`1 / ppf`. Exports use the first prepared record and only apply a floor when
explicitly requested. The noise-floor column gives the
exported contrast divided by `ppf`. Sky transmission is independent of the aperture.

## Example data

A small set of reference YIPs ships as GitHub release assets on this repo
and can be fetched on demand. This is useful for tutorials, CI, and "I just
want to try it" exploration:

```python
from yippy import open_yip

package = open_yip("eac1_aavc_2d")  # Download and cache the catalog entry.
coro = package.prepare()
```

The catalog is currently minimal. Long-term YIP hosting will be provided
by ExEP, and only a handful of reference YIPs from the Coronagraph Design
Survey are shipped here: `eac1_aavc_2d` and `eac1_optimal_order_6_1d`
(used by the yippy paper validation pipeline and as the pyEDITH default
coronagraph) and `usort_offaxis_ovc` (an off-axis vortex coronagraph
design study). For production work or YIPs outside this set, manage your
own YIP paths and load them with `open_yip(path)`.

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
from yippy import fetch_yip

# Per-call override (wins over the env var)
yip_path = fetch_yip("eac1_aavc_2d", cache_path="/data/shared/yips")

# Introspect the resolved cache directory without triggering a fetch
from yippy import cache_dir

print(cache_dir())
```

## Precision and execution

Enable JAX x64 before constructing a model when deep-contrast accuracy or the
sampled-intensity reader requires double precision:

```python notest
import jax
from yippy import open_yip

jax.config.update("jax_enable_x64", True)
coro = open_yip("path/to/yip").prepare()
```

There is no per-object dtype option. Datacube caches distinguish float32 and
float64. Model construction and export perform host I/O; image and scalar
queries support JAX transformations. Batch source positions through
`source_psfs(None, x_lod_array, y_lod_array, grid=target)`.

### Sampled-intensity reader

For a directory with a `sinc-delivery-0.1` manifest, enable JAX x64 before
preparing the sampled-intensity reader.

```python notest
import jax
from yippy import open_yip

jax.config.update("jax_enable_x64", True)
package = open_yip("delivery", product_id="mono-000")
coro = package.prepare(interpolation="sinc")
image, valid = coro.source_psfs(None, [4.0], [5.5], grid=None)
```

`blend` remains the default for unmarked packages.
Unmarked packages retain their numerical and cache behavior. `auto` records its
construction decision in the frozen `interpolation_info` record. It can use
blend for supported sampling refusals on full signed grids. Quadrant deliveries
require validated reflection evidence and midpoint detector geometry; they have
no blend fallback. Explicit `blend` rejects these deliveries. A bounded policy
never constructs a blend backend, including for unmarked packages.

For `sinc-delivery-0.1`, FITS `PIXSCALE` with an empty or absent comment uses
the schema unit, lambda0/D per pixel, and is compared numerically with the
manifest. Explicit lambda/D or lod units also use that frame; recognized arcsec,
mas, deg or rad units are converted before comparison. Nonempty unrecognized
comments raise. All declarations, referenced hashes and independent-image indices
are validated before precision or memory fallback decisions.
Absent physical admission claims produce `METADATA_MISSING`; geometry and
output-unit omissions remain data errors. A failed non-null campaign sampling
minimum requires a revised manifest removing that claim. The adapter evaluates
that campaign test with the core's separate binary64 nm-frame arithmetic.

Images contain incident photon fractions per pixel. Integral samples receive
only their declared photon conversion; mean densities additionally receive the
pixel area. Sinc output retains signed excursions. Each source read returns an
image and a validity flag. Refused positions carry NaN pixels in eager and
compiled reads.

The model selects one monochromatic product and reads stellar and sky data
from the manifest's separate package. Each `Image` carries its own `PixelGrid`
and wavelength/diameter `Frame`. `source_psfs`, `stellar_map`, and
`uniform_sky_map` accept explicit target grids and reconcile different
frames through angular coordinates. `grid=None` retains the native pixel grid
for the interpolated image. For a multiband instrument, select the imaging
model with `multi.band(wavelength_nm)`.

`SincOptions` accepts sinchole domain, memory and policy objects, plus an
optional prefilter `taps` count (5 to 10). By default sinchole selects its
7-tap prefilter for sampling ratios up to 0.64 and the raw kernel up to 0.76.
Prefilter tone-error estimates are not certified image-error bounds.
yippy retains raw training samples for performance diagnostics and includes
them in the prefiltered memory budget. Domain overrides
can only restrict the declared served geometry. Policy error budgets use photon
fractions per pixel. The default memory budget is computed from the actual
native cube size using sinchole's admission and 1024-output estimates, one-query
chunks and 256-pixel tiles. It excludes compiler workspace and caller-owned
arrays; larger batches need an explicit budget. Cube caches preserve per-query
status and backend selections in an integrity-checked sidecar. Every opt-in
cache identity retains the requested domain, policy and canonical options,
including when construction binds blend.

The `interpolation` selector accepts `blend` (default), `auto`, and `sinc`.
