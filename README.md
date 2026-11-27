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
throughput = record.evaluate("throughput", 5.0)
contrast = record.evaluate("raw_contrast", 5.0, 0.0)
area = record.evaluate("core_area_lod2", 5.0)
sky = coro.uniform_sky_response(5.0)
```

`package.prepare(...)` and `yippy.Coronagraph(package, ...)` build the same
model. `open_yip` returns a `YIPPackage`. It reads the FITS headers and hashes
the source files, and decodes the image arrays on first access. The read-only arrays preserve sample order,
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
with a different set of prepared rules, without loading new tables.

`leakage_rule` sets how the star's leakage is read at each separation. Every
curve is a function of separation alone, so on a 2D product (a stellar map that is
not rotationally symmetric) the leakage is an average over position angle or a
single cut, never the value at a planet's own angle; that value comes from the
image or aperture-record route of coronagraphoto. The rules are:

- `"aperture_ring_mean"`: the default for a 2D product. At each separation, the
  exact-coverage aperture of the rule (same circle, same pixel weights as the
  throughput) averaged over position angle, `(1/2pi) integral of the aperture sum
  with centre (s cos(phi), s sin(phi)) over phi`. It is computed as the aperture
  weights summed over the stellar map replaced by its ring means, which is the same
  quantity because the angle average of an aperture is rotationally symmetric. Each
  ring mean uses `ceil(2 pi r_max)` equally spaced angles (1131 for a 256 pixel
  map), one per pixel at the farthest ring; the ring means change by under 1e-4
  between 256 and 1024 angles on the shipped `eac1_aavc_2d` map.
- `"aperture_mean"`: the default for a 1D product. The aperture on the +x axis, the
  same value at every angle of a rotationally symmetric product. On a 2D product it
  is a +x cut, neither the planet value nor the azimuthal mean.
- `"azimuthal_ring_mean"`: deprecated. A point-sample ring mean times the aperture
  area, which ignores the aperture extent; `"aperture_ring_mean"` replaces it and
  agrees with it as the aperture shrinks to a point.

Pass the same `leakage_rule` to `curves` that the coronagraph was prepared with
(`curves` uses it by default). Reading a 2D product's curves logs one warning per
product saying so.
`bandpass` is a `optixstuff.spectral.Band`, or `None` when the YIP has no edges.
`primary` reads both diameter conventions and collecting area from the YIP.

## Images, grids, and JAX

Images carry array values in `.data` and geometry in `.grid`.
`source_psfs` returns an `Image` with leading axis `"source"` and a boolean
validity vector. A source is valid when it lies inside the source domain, the
hull of the tabulated off-axis nodes (62 lambda/D for `eac1_aavc_2d`, which is
wider than its 32 lambda/D PSF stamp); an invalid source has an all-zero image,
never NaN. Stellar and sky maps are two-dimensional for scalar diameters.
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
`yippy.lod_inscribed_to_circumscribed(separation_lod, frame, diameter_inscribed_m)`
converts offsets measured in inscribed-diameter lambda/D to the circumscribed
frame of a model.

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

For an extended scene, prepare its operator outside JIT. The render is always
finite: a lit source cell that no operator covers (outside the hull of the
tabulated source nodes, or outside every operator's mask) contributes zero and
sets `SceneRender.valid` to false.

```python notest
coro = coro.with_scene_operator()
render = coro.render_scene(scene_image, None, grid=detector_grid)
image, valid = render.image, render.valid
```

`with_scene_operator()` returns a `SceneCoronagraph`, which carries a tuple of
scene operators (`operators`) beside the unchanged model: a
`BlendSceneOperator` or a sinc `SceneRenderer`. Each operator is a linear map
from source flux to an image that renders the source cells it covers and
returns, with the image, a boolean mask of those cells. The render is the sum
of their images and is valid when every lit cell is covered by some operator.
`with_scene_operator(mask=...)` builds an operator over the cells of a mask
only; masked operators over disjoint masks add to the operator over their
union, and `SceneCoronagraph.with_operators(...)` combines them.
`MaskedSceneOperator` restricts any operator the same way.

The scene is a cell-integral image on `coro.scene_grid`: the stored PSF grid,
extended symmetrically to the hull of the tabulated off-axis nodes where they
reach beyond it (496 by 496 cells at 0.25 lambda/D for `eac1_aavc_2d`, whose
nodes reach 62 lambda/D, against its 256 by 256 stamp), so a source between the
stamp edge and the outermost node is rendered. The image is on `coro.grid`.
The blend operator renders the scene with one FFT product per stored PSF, so
it holds the stored PSFs rather than one PSF per source pixel (about 0.12 GB in
float32 for `eac1_aavc_2d` on its scene grid, where a PSF per stamp-grid source
takes 8.6 GB), and it is exactly linear in the source flux. A render may ring
slightly below zero, so clip it where a nonnegative image is required. A stored
PSF contributes only inside its own array translated to the source, in the
render and in `source_psfs` alike, so the render of one lit cell clipped at
zero is the PSF read of that source.

Build a disk on `coro.scene_grid`. A uniform disk filled exactly to the edge of
that grid still reads `valid` False, because the outermost cells lie just
outside the polygon of the off-axis nodes and their flux is dropped; keep lit
flux inside the node hull to render with `valid` True.

`coro.scene_operator_bytes()` reports the resident and scratch bytes of the
operator from shapes alone. `with_scene_operator(max_bytes=2 * 2**30)` raises
`hwoutils.memory.MemoryLimit` (a `ValueError`) before allocating when the
estimate exceeds `max_bytes`, on every route including prefiltered products;
pass a larger value, or `None` to skip the check. Prefiltered and per-source
routes read the stored PSF grid.

## Performance records

`coro.curves(rule)` returns an `optixstuff.PerformanceCurves` containing
`separation_lod`, `throughput`, `core_area_lod2`, `leakage_fraction`,
`leakage_density_lod2`, `raw_contrast`, `iwa_lod`, and `data_extent_lod`.
Each record carries the coronagraph's uniform-sky profile in `record.uniform_sky`.
The corresponding variance moments are `throughput_variance`,
`core_area_variance_lod2`, `leakage_fraction_variance`, and
`leakage_density_variance_lod2`.
Use `record.evaluate(field, separation_lod)` to interpolate; the leakage and
contrast fields take the star's diameter in lambda/D as a third argument,
`record.evaluate("leakage_density_lod2", separation_lod, 0.0)`, because the
record holds one leakage profile per stellar diameter of the product
(`stellar_diameters_lod`). Throughput and
area use linear-value PCHIP; leakage and contrast use log-value PCHIP. The
fields interpolate independently. Their contrast identity holds at measured
knots: contrast = leakage_density * core_area / throughput.

`Circular(radius_lod, centre="source")` fixes the aperture at the supplied
source position. `centre="peak"` follows the local PSF peak. `HalfMaximum(f)`
selects the connected core above fraction `f` of its peak. Both integrate
planet and stellar intensity over the same region and have identical mean
and variance moments.

Default records measure the shipped non-negative off-axis samples. For
standalone aperture measurements on another separation axis, use
`optixstuff.performance.performance_curves`. Its source callback returns a
native two-dimensional `Image`; select the source axis from `source_psfs(...).image`
with `dataclasses.replace`, retaining the native grid and reconstruction.
The prepared `coro.curves(rule)` records require no callback.

Records describe the point-source stellar map. Finite stellar diameters are an
image capability: `coro.stellar_map(None, diameter_lod, grid=None)` returns the
interpolated finite-star image.

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
from yippy.export import export_ayo_csv, export_exosims

ppf = 30.0
record = coro.curves()
contrast_noise = (
    np.maximum(np.abs(record.evaluate("raw_contrast", 5.0, 0.0)), 4e-10) / ppf
)
density_noise = record.evaluate("leakage_density_lod2", 5.0, 0.0) / ppf
export_exosims(coro)
export_ayo_csv(coro, "output.csv", ppf)
```

`optixstuff.PostProcessing.speckle_residual_fraction` owns the residual fraction
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
image and a validity flag. `source_psfs` gives a refused position an all-zero
image and `valid` False, in eager and compiled reads; the sinc reader's own
`read` returns NaN pixels with a refused status.

The model selects one monochromatic product and reads stellar and sky data
from the manifest's separate package. Each `Image` carries its own `PixelGrid`
and wavelength/diameter `Frame`. `source_psfs`, `stellar_map`, and
`uniform_sky_map` accept explicit target grids and reconcile different
frames through angular coordinates. `grid=None` retains the native pixel grid
for the interpolated image. For a multiband instrument, select the imaging
model with `multi.coronagraph_at(wavelength_nm)`.

`SincOptions` accepts `yippy.sinc` domain, memory and policy objects, plus an
optional prefilter `taps` count (5 to 10). By default `yippy.sinc` selects its
7-tap prefilter for sampling ratios up to 0.64 and the raw kernel up to 0.76.
Prefilter tone-error estimates are not certified image-error bounds.
yippy retains raw training samples for performance diagnostics and includes
them in the prefiltered memory budget. Domain overrides
can only restrict the declared served geometry. Policy error budgets use photon
fractions per pixel. The default memory budget is computed from the actual
native cube size using the `yippy.sinc` admission and 1024-output estimates, one-query
chunks and 256-pixel tiles. It excludes compiler workspace and caller-owned
arrays; larger batches need an explicit budget. Cube caches preserve per-query
status and backend selections in an integrity-checked sidecar. Every opt-in
cache identity retains the requested domain, policy and canonical options,
including when construction binds blend.

The `interpolation` selector accepts `blend` (default), `auto`, and `sinc`.
