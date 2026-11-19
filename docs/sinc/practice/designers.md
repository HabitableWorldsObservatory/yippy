---
file_format: mystnb
kernelspec:
  name: python3
---

# What must a designer deliver so that reads between the nodes are exact?

This page is for the coronagraph designer who propagates the PSFs. A
delivery that follows it can be admitted by yippy.sinc and read through yippy
without any change on the receiving side, and a delivery that misses a step
is refused with a reason rather than read incorrectly. The physics behind
each step is on the theory pages, the file layout is on {doc}`yippy` and
yippy's [schema page][schema], and {doc}`delivery-recipe` shows how to do
each step in order.

:::{admonition} The question of this page
:class: important

Which images do you propagate, in which frame and units, and under which
symmetry argument, so that the library can read the PSF at positions you
never propagated? The sections below answer it one object at a time: the
spacing of the nodes, the rectangle of nodes, the frame every image shares,
and the symmetry that lets one quadrant stand for the whole field. The close
lists what to tick off before sending.
:::

```{code-cell} python
:tags: [remove-cell]
import sys

sys.path.insert(0, "../scripts")
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown
from myst_nb import glue
from validation_tables import load

import yippy.sinc as sc
from yippy.sinc import viz
import toy_coronagraph as tc

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")


def tex(value, digits=2):
    """Format a number as inline math a x 10^b."""
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    return Markdown(rf"${mantissa}\times10^{{{int(exponent)}}}$")
```

## How far apart may the nodes be?

The delivery is a grid of images, one for each source position on a regular
lattice of nodes. The spacing of that lattice is the first thing you choose,
and it is not free: if the nodes are too far apart, the reads between them
cannot be exact, however good the images are. The figure draws the five
quantities that fix the spacing, for the public starter request, from left
to right. Each of the five panels adds one quantity to the previous one:

1. The circle is the pupil, whose illuminated width is the support width $L$
   in units of $D$.
2. The triangle is the sketch of the pupil's autocorrelation. Its half-width
   is the bandwidth $B=\sigma_{\max}L$, where $\sigma_{\max}$ is the
   reference wavelength divided by the shortest contributing wavelength
   ({doc}`../theory/band-limits` derives it).
3. The two rows of dots compare the spacing $h$ you may use (cyan) with the
   Nyquist spacing (gray). The cyan dots are closer by the factor
   $\rho$, the oversampling factor of {doc}`../foundations/sampling`.
4. The count is the nodes per axis that cover the region you will serve,
   with 16 more for the stencil.
5. The last panel is the lattice that results, with the box that the
   complete stencils serve.

```{code-cell} python
:tags: [hide-input]
from optixstuff.grids import Frame
fig, ax = plt.subplots(figsize=(10, 2.6), constrained_layout=True)
viz.plot_spacing_rule(
    lattice=sc.source_grid(jnp.zeros(2), jnp.full(2, 0.38), (31, 27), Frame(500.0, 6.0, "circumscribed")),
    support_width=1.0,
    ax=ax,
)
plt.show()
```

The arithmetic behind panel 3 is one division. The largest $\rho$ that the
32-tap kernel tolerates is 0.76 ({doc}`../theory/kernel` shows the error
rising steeply beyond it), and $\rho=2Bh$, so the largest spacing is
$h=0.76/(2B)$. For $L=1$ at $\lambda_0$ we have $\sigma_{\max}=1$ and $B=1$,
which gives $h=0.76/2=0.38$. A support narrower than $D$ leaves room: the
starter declares 0.946 by 0.982 $D$, so it sits below the limit on both axes.

```{code-cell} python
floor = load("floor_diagnosis.json")["band"]["declared_half_band"]
declared = np.array([floor["x"], floor["y"]])  # illuminated widths, in D
sigma_max = 1.0  # reference wavelength / shortest wavelength
margin, spacing = 0.76, 0.76 / (2 * sigma_max * 1.0)
rho = 2 * sigma_max * declared * spacing
print(f"plan: h = {margin} / (2 * 1 * 1) = {spacing}")
print(f"starter declaration {declared} D: rho = 2 * L * h = {np.round(rho, 3)}")
glue("h_plan", f"{spacing:.2f}", display=False)
glue("rho_x", f"{rho[0]:.3f}", display=False)
glue("rho_y", f"{rho[1]:.3f}", display=False)
```

So the plan uses $h=$ {glue:text}`h_plan` $\lambda_0/D$ and the declared
widths give $\rho=$ {glue:text}`rho_x` along $x$ and {glue:text}`rho_y` along
$y$, both below 0.76. Using the larger spacing for a wider support, or a
shorter wavelength than you declared, raises $\rho$ and the admission is
refused with `KERNEL_MARGIN_FAIL`. The label $27\times31$ counts nodes along
$x$ and then along $y$, while the lattice shape is ordered
$(n_y, n_x)=(31, 27)$.

Declare the support widths and the shortest wavelength, and do not round the
spacing after propagating: the nodes must sit at exactly the declared
multiples of $h$.

## Which nodes must be stored?

The last panel of the figure showed the stored quadrant and a larger box that
it serves. The next figure draws the same starter lattice on its own, with
one read marked. A read at a position between nodes combines the 32 by 32
nodes around it, a stencil ({doc}`../foundations/sinc-interpolation`), so a
read near the edge of the served region needs nodes beyond that edge,
including nodes you would never plan to use. The cell finds the stencil of a
read at $(4.0, 5.5)$.

```{code-cell} python
h = 0.38
query = np.array([4.0, 5.5])
below = np.floor(query / h).astype(int)  # the node just below the query
first, last = below - 15, below + 16  # 16 nodes each side
print(f"query / h = {np.round(query / h, 2)}, node just below = {below}")
print(f"stencil nodes: x {first[0]} to {last[0]}, y {first[1]} to {last[1]}")
print(f"mirrored columns: {max(0, -first[0])}, mirrored rows: {max(0, -first[1])}")
glue("mir_cols", str(max(0, -first[0])), display=False)
glue("mir_rows", str(max(0, -first[1])), display=False)
glue("last_col", str(last[0]), display=False)
glue("last_row", str(last[1]), display=False)
```

The stencil runs from node $-5$ to node 26 along $x$ and from $-1$ to 30
along $y$. The negative indices are the mirrored nodes, which exist only by
symmetry, and the highest indices are the last stored column and row, so the
starter needs 27 columns and 31 rows and no more. In general a quadrant
served up to, but not including, $R$ along an axis needs $\lceil R/h\rceil+16$
nodes there: $\lceil4.18/0.38\rceil+16=27$ and $\lceil5.7/0.38\rceil+16=31$.

The next figure shows the starter request. Look for the outlined rectangle
around the white cross: it is the stencil of the read above, with
{glue:text}`mir_cols` mirrored columns at negative $x$ and {glue:text}`mir_rows`
mirrored row at negative $y$, and its right and top edges fall on the last
stored column and row. The cyan dots are the stored nodes and the gray
circles are the mirrored nodes that exist only by symmetry.

The shaded square
is the stencil coverage: the region where every read has a complete stencil
among the stored and mirrored nodes, so reads are supported everywhere in it.
The cell the delivery declares as served and qualifies,
$3.8\le s_x<4.18$ and $5.32\le s_y<5.70$, is a small box at the cross, too
small to see at this scale. A delivery may declare the whole shaded square, in
which case reads outside the small cell would be supported but unqualified.

```{code-cell} python
:tags: [hide-input]
from optixstuff.grids import Frame
starter = sc.source_grid(jnp.zeros(2), jnp.full(2, 0.38), (31, 27), Frame(500.0, 6.0, "circumscribed"))
cell = sc.Domain("box", "lower_closed_upper_open",
                 jnp.array([[3.8, 5.32], [4.18, 5.70]]), jnp.zeros(2))
fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
viz.plot_coverage(starter, domain=cell, quadrant=True, query=(4.0, 5.5), ax=ax)
plt.show()
```

Deliver the complete rectangle, including nodes inside the occulter or
outside the useful region, and keep each image tied to the integer indices
$(i_x, i_y)$ of its node: the source position is $(h\,i_x, h\,i_y)$, starting
exactly at the origin for a quadrant. A missing node is refused with
`GRID_INCOMPLETE`, and a served region that reaches past the stored nodes
with `INCOMPLETE_STENCIL`.

## What must be the same in every image?

Every image shares one detector frame: the same shape, the same pixel pitch
and the same center, with the star at that center. For a quadrant the center
is `((nx - 1)/2, (ny - 1)/2)` in zero-based pixels, which is between the two
middle pixels for an even dimension and on the middle pixel for an odd one.
Increasing column follows positive source $x$ and increasing row follows
positive source $y$. A quadrant whose center is anywhere else is refused with
`FRAME_UNSUPPORTED`, and the recipe measures it
({ref}`step 2 <recipe-frame>`).

The units must not depend on the source. Each value is a pixel integral, or a
declared mean density, normalized to incident photons at a declared plane, and
no individual PSF is normalized to its own sum or peak, because a factor that
changes with source position changes the image's bandwidth
({doc}`../theory/band-limits`). The yippy format of version 0.1 accepts the
integral and the mean but not point samples, which yippy.sinc alone accepts.
The recipe's {ref}`step 3 <recipe-pixel>` shows the three measures.

## When does one quadrant serve the whole field?

The theory page ({doc}`../theory/quadrant-symmetry`) explains why a real
pupil, a real Lyot stop and a real, centrosymmetric occulter make the
intensity unchanged when source and detector are inverted together, which is
conjugation, so one geometric mirror is then enough for both axes. The toy
coronagraph tests this by decentering its pupil. A shift of
{glue:text}`shift_pct` percent of $D$ along $x$ keeps the pupil symmetric about
the $x$ axis, and an equal shift along both axes leaves it symmetric about
neither.

What to look at in the next figure: each panel has three rails, and the
middle rail holds the three mirror residuals (the $x$-negative, $y$-negative
and both-negative checks), each as a fraction of the local peak. The dashed
line on that rail is the limit, {glue}`mirror_tol`. In the left panel every
residual sits at rounding level, far below the line. In the right panel two
residuals sit far above it. The toy delivery carries no interior checks, so
the top rail is empty, and the two high residuals of the diagonal case are
equal and overlap. The bottom rail compares each stored axis image with its
own flip, and in the right panel those scatter above the limit as well.

```{code-cell} python
:tags: [hide-input]
cases = {"pupil shifted along x": (0.02, 0.0), "pupil shifted diagonally": (0.02, 0.02)}
receipts = {}
for name, shift in cases.items():
    delivery = tc.quadrant_delivery(tc.ToyCoronagraph(pupil_shift=shift))
    admission = sc.admit_grid(**delivery)
    receipts[name] = admission.receipt
    print(f"{name}: admitted={admission.admitted}, "
          f"reasons={sc.reason_names(admission.reason_bits)}")
    print("   mirror residuals (x-neg, y-neg, both-neg):",
          ", ".join(f"{v:.1e}" for v in admission.receipt.mirror_local_peak))
diagonal = receipts["pupil shifted diagonally"].mirror_local_peak
glue("shift_pct", f"{100 * cases['pupil shifted along x'][0]:.0f}", display=False)
glue("diag_pct", f"{100 * float(diagonal[0]):.1f}", display=False)
glue("mirror_tol", tex(float(delivery["reflection"].tolerance), 1), display=False)
glue("x_round", tex(float(max(receipts["pupil shifted along x"].mirror_local_peak))), display=False)
```

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True, constrained_layout=True)
for ax, (name, receipt) in zip(axes, receipts.items()):
    viz.plot_check_residuals(receipt, ax=ax)
    ax.set_title(name)
plt.show()
```

The shift along $x$ passes every check, at most {glue}`x_round` of the local
peak, which is float64 rounding: the pupil keeps its $x$-axis mirror and
conjugation supplies the other. The diagonal shift fails the $x$-negative and
$y$-negative checks at {glue:text}`diag_pct` percent of the local peak, and
admission refuses with `SYMMETRY_CHECK_FAILED`. Its both-negative check still
passes, because inverting both coordinates is exactly what conjugation
provides, so the three checks together separate the two arguments. A delivery
that fails them needs a full signed grid, not a looser tolerance: yippy.sinc
accepts no tolerance above {glue}`mirror_tol`. The families that need a full
grid, and the measurement that a complex mask fails the check, are in
{ref}`step 5 of the recipe <recipe-symmetry>`.

The tolerance is relative to the peak of the planet core in each image, so
the check tests the core and the near field. It cannot see an asymmetry
smaller than that fraction of the local peak, for example in the far wings,
and how small an asymmetry the science can tolerate is not something the
library states. Admission therefore establishes the symmetry of the core and
the near field, and the designer's declared argument carries the rest.

## What to tick off before sending

Each row is one thing to deliver, the reason the library gives when it is
missing or wrong, and the section of the recipe that shows how. The last two
rows are not refused by admission and are reported with the files instead.

```{list-table}
:header-rows: 1
:widths: 4 46 24 26

* -
  - Deliver
  - Refused with
  - How
* - 1
  - A spacing $h\le0.76/(2\sigma_{\max}L)$ on each axis, from the full
    illuminated widths and the shortest contributing wavelength
  - `KERNEL_MARGIN_FAIL`
  - {ref}`Step 1 <recipe-positions>`
* - 2
  - The integer index of every image, with positions $(h\,i_x, h\,i_y)$
    starting at the origin
  - `GRID_NONUNIFORM`
  - {ref}`Step 1 <recipe-positions>`
* - 3
  - The complete rectangle of $\lceil R/h\rceil+16$ nodes per axis, including
    nodes inside the occulter
  - `GRID_INCOMPLETE`, `INCOMPLETE_STENCIL`
  - {ref}`Step 1 <recipe-positions>`
* - 4
  - One frame for every image, the star at the center
  - `FRAME_UNSUPPORTED`
  - {ref}`Step 2 <recipe-frame>`
* - 5
  - Source-independent units: a pixel integral or declared mean density, in
    photons at a declared plane
  - `NORMALIZATION_UNSUPPORTED`
  - {ref}`Step 3 <recipe-pixel>`
* - 6
  - A symmetry argument for the optical state, and $x$-negative, $y$-negative
    and both-negative images propagated directly
  - `SYMMETRY_CHECK_FAILED`, `SYMMETRY_CHECK_MISSING`
  - {ref}`Step 5 <recipe-symmetry>`
* - 7
  - Interior checks: images between nodes, in the same frame, state and
    units
  - none (evidence only)
  - {ref}`Step 1 <recipe-positions>`
* - 8
  - Convergence at refined settings, and float32 against float64 storage
  - none (reported)
  - {ref}`Step 6 <recipe-check>`
```

Also check that the manifest lists every file with its SHA-256 hash and that
the FITS headers agree with it (`PIXSCALE` in $\lambda_0/D$ per pixel, with the
unit in its comment), that images are finite and non-negative, that stellar and
sky products stay separate and unchanged, and that failed or unperformed
checks are reported as such, never as passing. Run `yippy-sinc validate` on the
directory, and check that a fiducial source at $(s,0)$ has its core at column
$c+s/\text{PIXSCALE}$.

## What this page established

The spacing follows from the support and the shortest wavelength, the stored
rectangle follows from the spacing and the 32-node stencil, every image
shares one frame and source-independent units, and one quadrant serves the
whole field only under a symmetry argument that the mirror checks test. The
next page, {doc}`declarations`, lists what each of these claims in the
files.

Next: {doc}`declarations`.

[schema]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html
