# Available YIPs and Downloading

The primary way to use yippy is to point it at a YIP directory you already
have on disk -- either one you built yourself or one provided by a
coronagraph designer:

This example needs a user-supplied YIP directory.

```python notest
from yippy import open_yip

package = open_yip("path/to/eac1_aavc_2d")
coro = package.prepare()
```

For convenience, yippy also ships a small curated catalog of YIPs that
can be downloaded on demand via [pooch](https://www.fatiando.org/pooch/).
The catalog is hosted as assets on a tagged GitHub release of the yippy
repository and cached locally on first use.

```{note}
The catalog currently includes a handful of reference YIPs from the
Coronagraph Design Survey. Long-term YIP hosting will be provided by
ExEP, and the discovery API here will grow back into a thin proxy over
that catalog when it comes online. For production work or YIPs outside
this set, manage your own YIP paths and pass them to
`open_yip(path)`.
```

## Available datasets

The table below lists the releases available through `list_yips()`. Each
archive has a pinned digest. Call `yip_info(name)` to inspect its release,
download URL and cache location; it returns an `Archive` record.

```{include} _generated/yip_catalog.md
```

### Design references

Cite the coronagraph design, not just yippy, when you publish results
built on one of these YIPs:

- `eac1_aavc_2d` -- the EAC1 apodized vortex coronagraph designed by
  Susan Redmond. D. P. Mawet, S. F. Redmond, A. Bertrou-Cantou,
  G. Ruane, and J. Llop-Sayson, "Apodized vortex coronagraph for the
  Habitable Worlds Observatory," in *Techniques and Instrumentation for
  Detection of Exoplanets XII*, Proc. SPIE **13627** (2025),
  [doi:10.1117/12.3065779](https://doi.org/10.1117/12.3065779).

## Sampling

YIP names end with a sampling-regime suffix:

- `_1d` -- radially-symmetric sampling. The PSF cube is indexed by radial
  separation; lighter weight, suitable for back-of-envelope yield work.
  Off-axis PSFs are translated to the requested position, not rotated to
  its position angle, so azimuthal structure in the tabulated PSFs keeps
  its +x orientation. Not recommended for spatially-resolved image
  simulation.
- `_2d` -- full 2D off-axis grid. Captures azimuthal variation; required
  whenever PSF asymmetries matter (e.g. for spatially-resolved benchmarks).

## Downloading a YIP via `fetch_yip`

Three call styles, all equivalent for clean cases:

```python
from yippy import fetch_yip, open_yip

# Flat name (best when you already know it)
path = fetch_yip("eac1_aavc_2d")

# Structured query (must resolve to a single entry). Pass sampling
# whenever a (telescope, coronagraph) pair has both 1D and 2D variants.
path = fetch_yip(telescope="eac1", coronagraph="aavc", sampling="2d")

# Convenient: top-level re-exports
import yippy
path = yippy.fetch_yip("eac1_aavc_2d")

# Then load it
coro = open_yip(path).prepare()
```

The first call downloads and unpacks the archive into the pooch cache
(roughly 30 MB per 1D YIP, ~75 MB for the 2D AAVC). Subsequent calls hit
the cache and are instantaneous.

## Cache location

By default, downloaded YIPs land in `pooch.os_cache("yippy")` -- the
OS-conventional cache directory provided by
[platformdirs](https://platformdirs.readthedocs.io):

| OS | Default location |
|---|---|
| macOS | `~/Library/Caches/yippy/` |
| Linux | `~/.cache/yippy/` (or `$XDG_CACHE_HOME/yippy/`) |
| Windows | `C:\Users\<user>\AppData\Local\yippy\Cache\` |

### Overriding the location

There are two override mechanisms, in priority order:

1. **`cache_path` keyword** -- per-call override. Wins over everything else.

   This example writes to a user-configured shared cache directory.

   ```python notest
   path = fetch_yip("eac1_aavc_2d", cache_path="/data/shared/yips")
   ```

2. **`YIPPY_CACHE_DIR` environment variable** -- persistent override.
   Any process (yippy, pyEDITH, your scripts) inherits the same cache
   location. Pick the snippet that matches your setup; the
   directory is created automatically on first download.

   **macOS / Linux (zsh, the default on modern macOS):**

   ```bash
   # Just this session -- run in the terminal you'll use yippy from:
   export YIPPY_CACHE_DIR=~/Documents/YIPs

   # Persistent -- add the same line to ~/.zshrc, then restart the
   # terminal (or run `source ~/.zshrc`) to pick it up:
   echo 'export YIPPY_CACHE_DIR=~/Documents/YIPs' >> ~/.zshrc
   ```

   If you use bash instead of zsh, swap `.zshrc` for `.bashrc` (Linux)
   or `.bash_profile` (older macOS). You can check which shell you're
   running with `echo $SHELL`.

   **Windows (PowerShell):**

   ```powershell
   # Just this session:
   $env:YIPPY_CACHE_DIR = "$HOME\Documents\YIPs"

   # Persistent (applies to all future sessions for your user):
   [Environment]::SetEnvironmentVariable(
       "YIPPY_CACHE_DIR", "$HOME\Documents\YIPs", "User"
   )
   ```

   **From inside a Python script** (affects lookups in this process):

   This example sets a user-specific cache directory.

   ```python notest
   import os
   from yippy import yip_info

   os.environ["YIPPY_CACHE_DIR"] = "/path/to/YIPs"
   print(yip_info("eac1_aavc_2d").archive_path())
   ```

   To confirm the variable is set, run `echo $YIPPY_CACHE_DIR` in your
   terminal -- it should print the path you set. If it prints a blank
   line, the variable is not set in that shell session.

If neither is set, yippy uses the platform default.

### Inspecting the resolved cache

```python
from yippy import cache_dir

print(cache_dir())
```

`cache_dir()` honors `YIPPY_CACHE_DIR` so it reflects the location the
next `fetch_yip` call (without an explicit `cache_path`) will use.

## Discovery helpers

Three small helpers let you browse the catalog without downloading anything.

### `list_yips` -- filter the catalog

```python
from yippy import list_yips

print(list_yips(telescope="eac1"))
# ['eac1_aavc_2d', 'eac1_optimal_order_6_1d']
print(list_yips(coronagraph="optimal_order_6"))
# ['eac1_optimal_order_6_1d']
print(list_yips(sampling="2d"))
# ['eac1_aavc_2d']
print(list_yips())  # everything
```

Valid filter keys are `telescope`, `coronagraph`, and `sampling`. Combine
any of them; the call returns names whose metadata matches every filter.

### `yip_exists` -- membership check

```python
from yippy import yip_exists

print(yip_exists("eac1_aavc_2d"))  # True
print(yip_exists("not_a_yip"))  # False
```

### `yip_info` -- inspect metadata

```python
from yippy import yip_info

archive = yip_info("eac1_aavc_2d")
print(archive.release, archive.digest)
print(archive.url)
print(archive.archive_path())  # Resolve the cache location without fetching.
```

Descriptive metadata beyond these fields (wavelengths, dark-zone extent,
pixel scale, ...) lives in each YIP's FITS headers, not in the catalog.

## What a coronagraph was built from

`Coronagraph.built_at` records the full SHA-256 digest of the delivery and the
selected product ID, when present. Its `stellar.`, `offaxis.` and `sky.` labels
record each product's design, native frame, jitter in milliarcseconds, wavefront
error in picometers, pupil geometry and declared pupil identifier. `producer_version`
records the yippy release that built it. Absent or unconvertible declarations
are `unknown`. An explicit zero for jitter or wavefront error is a known value.

Use `compare_stamps` to check which build facts agree, which differ and which
could not be verified. The stamps describe the delivery at build time. Changing
prepared aperture rules or replacing numerical operands keeps those facts.

```python
from hwoutils.provenance import compare_stamps
from yippy import open_yip

package = open_yip("eac1_aavc_2d")
first = package.prepare()
second = package.prepare()

for label, value in first.built_at:
    if label.startswith("stellar."):
        print(label, value)

print(compare_stamps(first.built_at, second.built_at))
```

## Naming the pupil a delivery was computed with

A pupil identifier is a name the producer of a delivery gives to the pupil
prescription its tables were computed with, for example the name of an aperture
file or of a design revision. yippy never derives one from the diameter or the
obscuration, because two pupils of equal extent are different prescriptions.

A producer writes the name with `yip_header(..., pupil_id="name")`, which adds
the `PUPIL_ID` card, or as the optional `pupil_id` key of a versioned delivery
manifest. A user who knows the provenance of an unlabeled delivery passes
`pupil_id=` to `YIPPackage.prepare`. The name must be a non-empty string, and
every declaration the package holds must agree: a keyword that contradicts a
card or a manifest key, or two products with different cards, raise a
`ValueError`. The identifier of the coronagraph is known when the keyword, the
manifest key, or the cards of all three image products declare it. A partial
declaration leaves `Coronagraph.pupil_id` as `None` and is kept in `built_at`
under the label of its product.

Installed in an `Observatory`, the identifier is compared with the one on the
telescope pupil. Equal known names give `pupil_match == "matched"`, two known
and different names raise `StampMismatchError` when the `Observatory` is built,
and an unknown name on either side gives `"unverified"`.

```python
import tempfile

import numpy as np
import optixstuff as ox
from hwoutils.provenance import StampMismatchError
from optixstuff.grids import Frame, PixelGrid
from yippy import open_yip, write_yip
from yippy.writer import yip_header

grid = PixelGrid.focal(16, 0.25, Frame(500.0, 6.0, "circumscribed"))
y, x = np.mgrid[:16, :16] - 7.5
offsets = np.arange(0.0, 2.1, 0.5)
psfs = np.array([np.exp(-((x * 0.25 - o) ** 2 + (y * 0.25) ** 2)) for o in offsets])
header = yip_header(
    grid, inscribed_diameter_m=5.0, obscured=0.0, pupil_id="aperture-rev-a"
)
products = dict(
    grid=grid,
    offaxis_psfs=psfs,
    offsets_xy_lod=offsets,
    stellar_intensity=1e-9 * np.ones((2, 16, 16)),
    stellar_diameters_lod=[0.0, 0.1],
    sky_transmission=np.ones((16, 16)),
    headers=header,
)
coronagraph = open_yip(write_yip(tempfile.mkdtemp(), products)).prepare()
print(coronagraph.pupil_id)

detector = ox.Detector(
    pixel_scale_arcsec=0.01,
    shape=(10, 10),
    qe=0.9,
    dark_current_e_per_pixel_s=0.0,
    read_noise_e_rms=0.0,
    clock_induced_charge_e_per_pixel_frame=0.0,
    read_time_s=1.0,
    survival_fraction=1.0,
)
imager = ox.Instrument("imager", coronagraph=coronagraph, detector=detector)


def telescope(pupil_id):
    pupil = ox.SimplePupil(
        circumscribed_diameter_m=6.0, collecting_area_m2=28.0, pupil_id=pupil_id
    )
    return ox.Telescope(pupil)


for pupil_id in ("aperture-rev-a", None):
    observatory = ox.Observatory(telescope(pupil_id), (imager,))
    print(observatory.select("imager").pupil_match)

try:
    ox.Observatory(telescope("aperture-rev-b"), (imager,))
except StampMismatchError as error:
    print("refused:", type(error).__name__)
```
