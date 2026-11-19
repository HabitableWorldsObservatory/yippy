---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  render_markdown_format: myst
---

# How does a delivery reach yippy.sinc through yippy?

Most users never call yippy.sinc directly. A coronagraph designer sends a
directory of files, and a user wants off-axis PSFs from it at positions nobody
propagated. [yippy](https://yippy.readthedocs.io) opens the directory as a
`Coronagraph` whose off-axis PSFs come from yippy.sinc. The yippy package owns
everything about files (parsing, units, hashes, caches and the choice of
backend), and yippy.sinc owns the grid validation, the reads and their refusals.
This page shows the yippy side of that division.

:::{admonition} The question of this page
:class: important

What is in a delivery, what does yippy check before yippy.sinc sees it, and
which choices does a user make? The sections below take the directory, the
backend choice, the stellar products, the options and the checks in that
order, and the close fills in the answers.
:::

:::{note}
The sinc reads ship with yippy. Enable JAX 64-bit mode before
constructing a `Coronagraph` with sinc reads.
:::

## What is a delivery?

A delivery is a directory whose `sinc-delivery.json` manifest (schema
`sinc-delivery-0.1`) names every file and its SHA-256 hash. A **YIP** (yield
input package) is the older yippy format, a directory of FITS files for a
coronagraph's stellar intensity, sky transmission and off-axis PSFs. A
delivery adds a complete grid of off-axis PSFs and the checks that let
yippy.sinc admit it. The files of the public starter delivery are listed below,
each with what it holds.

```text
sinc-delivery.json                   manifest: frame, layout, symmetry, hashes
starter_training.csv                 indexed positive-quadrant positions
starter_checks.csv                   independent interior check positions
starter_mirror_checks.csv            independent negative-position checks
mono-000/offax_psf.fits              training images [source, row, column]
mono-000/offax_psf_offset_list.fits  offsets [source, 2], x then y
checks/mono-000.fits                 independent check images
mirror-checks/mono-000.fits          mirror check images
mirror-report.json                   negative-position and axis comparisons
legacy-yip/                          stellar and sky products, unchanged
                                     (needed by Coronagraph)
```

The training images are the nodes of the grid, one image per propagated
position. The independent checks are images at positions between the nodes,
propagated directly, which admission reads and compares. The mirror checks
are images at negative positions, which test the symmetry that lets the grid
store only the positive quadrant. The manifest holds everything that is not
an image: the detector frame, the pixel and normalization conventions, the
symmetry claim and the **served region**, the positions where the delivery
permits reads.

The files in this directory are the arguments of yippy.sinc's `admit_grid`. The
cell below builds the toy delivery of {doc}`first-read`, which has the same
structure at a smaller size, and prints the shape of each array that yippy
would load from the files above.

```{code-cell} python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import yippy.sinc as sc
import toy_coronagraph as tc

toy_delivery = tc.quadrant_delivery(tc.ToyCoronagraph())
rows = [
    ("intensity_nodes", toy_delivery["intensity_nodes"].shape,
     "offax_psf.fits, with a leading product axis"),
    ("offsets_xy_lod", toy_delivery["offsets_xy_lod"].shape, "offsets list, x then y"),
    ("indices_xy", toy_delivery["indices_xy"].shape, "integer indices from the training CSV"),
    ("mirror_checks", toy_delivery["mirror_checks"].intensity.shape,
     "mirror-checks/mono-000.fits"),
    ("lattice", toy_delivery["declared_lattice"].shape, "from the manifest layout"),
]
for name, shape, role in rows:
    print(f"{name:16s} {str(shape):18s} {role}")
print("admitted:", sc.admit_grid(**toy_delivery).admitted)
```

Each array is what yippy parses from a file and hands over unchanged, and the
declarations come from the manifest. The toy delivery passes no `checks`
argument, so it has no interior check images, and the starter has
{glue:text}`y_checks` of them plus {glue:text}`y_train` training images.

## Choosing the interpolation backend

`Coronagraph` takes an `interpolation` argument with three values.

`"blend"` (the default)
: The existing yippy interpolation of tabulated PSFs. Legacy YIP packages,
  their caches and their results are unchanged bit for bit.

`"sinc"`
: Strict yippy.sinc reads. The constructor raises if the delivery is not a
  versioned sinc delivery, if JAX 64-bit
  mode is off, or if admission refuses the grid.

`"auto"`
: The yippy.sinc reader serves reads when the delivery is a versioned sinc delivery and admission
  succeeds, otherwise the blend backend with the reason recorded in
  `coro.interpolation_info`. Data errors (a corrupted hash, a header that
  disagrees with the manifest, an unknown unit) always raise, even in auto
  mode, because they mean the files are wrong rather than that sinc is
  unavailable.

This example needs a delivery on disk.

```python notest
import jax

jax.config.update("jax_enable_x64", True)

from yippy import open_yip

coro = open_yip("path/to/delivery").prepare(interpolation="sinc")
image, valid = coro.source_psfs(None, [4.0], [5.5], grid=None)  # planet at (4.0, 5.5) lambda0/D
print(coro.interpolation_info.effective)  # "sinc"
print(coro.interpolation_info.receipt_id)  # the admission receipt identity
```

A quadrant delivery has no blend representation, so `"auto"` on a quadrant
delivery either reads with yippy.sinc or raises, and it never falls back.

## I already have a YIP: keep blend or request a sinc delivery?

```{code-cell} python
:tags: [remove-cell]
import sys

sys.path.insert(0, "../scripts")
from IPython.display import Markdown
from myst_nb import glue
from validation_tables import load, sci

v = load("validation.json")
glue("y_sinc", Markdown(sci(v["worst_check_error_local_peak"])), display=False)
glue("y_blend", f"{100 * v['blend_worst_error_local_peak']:.0f}", display=False)
glue("y_n", str(v["independent_checks"]), display=False)
glue("y_checks", str(v["independent_checks"]), display=False)
glue("y_train", str(v["training_nodes"]), display=False)
glue("y_mirror", str(v["mirror_checks"]), display=False)
```

An existing YIP package keeps working through `"blend"` with no change, so
the question is whether the science needs reads between nodes at a level that
blend cannot give. On the {glue:text}`y_n` interior checks of the public starter, the yippy.sinc
reads sit at {glue}`y_sinc` of the local peak at worst and the blend
reader on the same checks is off by {glue:text}`y_blend` percent by the same
measure, so the choice matters only for work that reads a planet at positions
nobody propagated.

```mermaid
flowchart TD
    A["I have a YIP package"] --> B{"Do you read planet PSFs<br/>between the stored nodes?"}
    B -- "no, or blend is accurate enough" --> C["Keep interpolation='blend'<br/>nothing changes"]
    B -- "yes, and it must be checkable" --> D["Request a sinc delivery<br/>from the designer"]
    D --> E["Designer plans positions with yippy-sinc plan<br/>and propagates the complete rectangle"]
    E --> F["Designer validates with yippy-sinc validate<br/>and sends the directory"]
    F --> G["open_yip(path).prepare(interpolation='sinc')"]
```

A YIP cannot be promoted by editing its files. A sinc delivery needs a
complete lattice at the planned spacing, pixel-integral images in a fixed
frame, and mirror and interior checks that were propagated directly, and a
YIP carries none of the three. What changes for the designer is the position
list and the checks. What does not change is the stellar and sky products:
they stay in their legacy YIP layout, unchanged, in the directory named by
the manifest key `legacy_package_path`.

## Why does `Coronagraph` need the stellar products?

`open_yip(path).prepare(interpolation="sinc")` builds the whole coronagraph
object, and the object also serves the stellar intensity and the sky
transmission, which yippy.sinc does not read. They come from the legacy
products (`stellar_intens.fits` and the others of a YIP package) in
`legacy_package_path`. A delivery meant for direct `Coronagraph` use must
therefore include them. The public starter holds none, so it is read through
yippy's router (`route_interpolation`), which returns the sinc backend
without them, and `yippy-sinc validate` checks its files either way. The
example below is not executed on this page, because the starter's release
asset is not published yet, and it follows the quickstart on yippy's schema
page.

```python notest
import jax

from yippy.datasets import fetch_sinc_starter
from yippy.sinc.routing import route_interpolation

jax.config.update("jax_enable_x64", True)

path = fetch_sinc_starter()
_, backend, info = route_interpolation(
    path,
    requested="sinc",  # strict: raise instead of falling back to blend
    options=None,  # None means the default SincOptions
    downsample_shape=None,  # None means no downsampling, the only choice for a delivery
)
psf = backend.create_psf(3.9425, 5.415)  # inside the declared corner cell
print(info.effective)  # "sinc"
```

The router returns three things: the parsed delivery (unused here), the
backend that serves reads, and the `InterpolationInfo` that records which
backend was chosen and why. The position lies in the corner cell that the
starter declares as served, and the next cell checks that with yippy.sinc's own
`Domain`, which is what the router builds from the manifest. The cell is
half-open: the lower edge belongs to it and the upper edge does not.

```{code-cell} python
corner = sc.Domain(
    "box", "lower_closed_upper_open",
    jnp.array([[3.8, 5.32], [4.18, 5.70]]), jnp.zeros(2),
)
for label, position in (
    ("the position above", (3.9425, 5.415)),
    ("on the lower edge", (3.8, 5.415)),
    ("on the upper edge", (4.18, 5.415)),
    ("one node below", (3.42, 5.415)),
):
    served = bool(corner.contains(jnp.array(position)))
    print(f"{label:20s} {position}  served: {served}")
```

A position outside that cell is refused for this delivery, although the stored
nodes could support it ({doc}`../validation`).

## Options

`SincOptions` selects a product and passes construction controls through to
yippy.sinc. The example narrows the permission region to a box inside the
starter's served cell.

This example needs a delivery on disk.

```python notest
from yippy import open_yip, SincOptions
import yippy.sinc as sc
import jax.numpy as jnp

options = SincOptions(
    product_id="mono-000",  # required for multi-plane deliveries
    domain=sc.Domain(  # optional narrower permission region
        "box", "closed", jnp.array([[3.85, 5.35], [4.1, 5.65]]), jnp.zeros(2)
    ),
)
coro = open_yip("path/to/delivery", product_id="mono-000").prepare(
    interpolation="sinc", sinc_options=options
)
```

A caller domain may narrow the delivery's served region and never widen it.
Delivered independent checks that fall outside the narrower region are
excluded, and the receipt lists them as excluded. A box such as
$[-4, 4]^2$ would be a mistake for the starter: it lies outside the corner
cell, so it would exclude every delivered check and serve nothing. Checks
outside the delivery's own served region are a data error. A bounded
`ReadPolicy` never falls back to blend: version 0.1 cannot issue bounds, so a
bounded request refuses.

The cell checks the narrow box against two positions, both inside the served
cell.

```{code-cell} python
narrow = sc.Domain(
    "box", "closed", jnp.array([[3.85, 5.35], [4.1, 5.65]]), jnp.zeros(2)
)
for label, position in (("inside both", (3.9425, 5.415)), ("served cell only", (3.81, 5.33))):
    print(f"{label:17s} {position}  served: {bool(corner.contains(jnp.array(position)))}"
          f"  in the narrow box: {bool(narrow.contains(jnp.array(position)))}")
```

## What yippy checks before routing

Every declaration and integrity check runs before yippy decides between sinc
and blend, so a damaged delivery cannot slip through a fallback.

The checks
cover the SHA-256 hash of every referenced file (images, offsets, coordinate
lists, independent checks, mirror checks and reports), agreement between FITS
headers and the manifest, the order of domain bounds, consistency of layout
and reflection, a complete rectangle and authoritative integer indices.
Missing physical metadata, such as the coordinate model or a premise, is
classified as missing (auto falls back with that reason, strict sinc raises),
while metadata that defines units or geometry (detector shape, center and
pitch, sample kind, normalization and photon conversion) is always a data
error.

## Units and frames

Off-axis images keep the delivered units, photon fraction per pixel for a
pixel-integral delivery, and yippy.sinc applies no normalization of its own.
Stellar and sky products keep their own frame and pixel scale, which may
differ from the off-axis frame. The `PIXSCALE` header is in reference
$\lambda_0/D$ per pixel. An empty header comment means that schema unit, a named
angular unit (arcsec, mas, deg, rad) is converted, and an unknown unit raises.

## Which commands check a delivery before it is sent?

Every manifest key, CSV column, FITS HDU and header keyword of the directory
above is defined on yippy's
[schema page][schema], and this page does not repeat them. The
designer-facing checklist is on {doc}`designers`, the step-by-step recipe is
on {doc}`delivery-recipe`, and the `yippy-sinc` command line has two
commands that a designer needs.

`yippy-sinc plan`
: Writes the training, check and mirror-check lists from the wavelengths,
  support and served limit.

`yippy-sinc validate <dir>`
: Runs the yippy integrity checks and then yippy.sinc admission with 64-bit
  arithmetic, and exits 0 on a pass, 1 on a failure and 2 on a usage error.
  The Python form is `yippy.sinc.tools.validate()`.

## What did this page establish?

The question was what a delivery holds, what yippy checks before yippy.sinc
sees it, and which choices a user makes. The answers, filled in:

- **A delivery** is a manifest plus the training images ({glue:text}`y_train`
  for the starter), independent check images ({glue:text}`y_checks`) and
  mirror check images ({glue:text}`y_mirror`), and the legacy stellar and sky
  products that `Coronagraph` needs.
- **yippy checks** every hash, header, layout and metadata declaration before
  it picks a backend, so a damaged delivery never reaches a fallback.
- **The user chooses** `"blend"`, `"sinc"` or `"auto"`, and a quadrant
  delivery never falls back to blend.
- **A caller domain** can only narrow the served region the delivery declares.

The next page asks what admission refuses and why.

Next: {doc}`admission`.

[schema]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html
