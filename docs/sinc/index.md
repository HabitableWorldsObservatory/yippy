---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  render_markdown_format: myst
---

# yippy.sinc

```{code-cell} python
:tags: [remove-cell]
import csv
import sys

sys.path.insert(0, "scripts")
from IPython.display import Markdown
from myst_nb import glue
from validation_tables import load, static_dir

import toy_coronagraph as tc

v = load("validation.json")
q = load("qualification_extended.json")
bm = load("benchmark.json")
with open(static_dir() / "qualification_extended.csv") as handle:
    rows = list(csv.DictReader(handle))
starter_rows = [r for r in rows if r["group"] == "starter"]
worst_row = max(starter_rows, key=lambda r: float(r["sinc_local_peak"]))


def tex(value):
    """Format a number as a LaTeX power of ten."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return Markdown(rf"${mantissa}\times10^{{{int(exponent)}}}$")


glue("n_reads", str(q["all"]["n"]), display=False)
glue("n_nodes", str(v["training_nodes"]), display=False)
glue("n_checks", str(v["independent_checks"]), display=False)
glue("n_mirror", str(v["mirror_checks"]), display=False)
glue("n_all_checks", str(v["independent_checks"] + v["mirror_checks"]), display=False)
glue("sinc_here", tex(float(worst_row["sinc_local_peak"])), display=False)
glue("blend_here", f"{100 * float(worst_row['blend_local_peak']):.0f}", display=False)
glue("blend_worst", f"{100 * v['blend_worst_error_local_peak']:.0f}", display=False)
glue("pixel_all", tex(q["all"]["sinc"]["pixel_brightest"]["worst"]), display=False)
glue("floor_tone", tex(tc.kernel_error_worst(0.76)), display=False)
glue("ms_f32", f"{1e3 * bm['float32']['single_steady_median_seconds']:.1f}", display=False)
glue("ms_f64", f"{1e3 * bm['float64']['single_steady_median_seconds']:.0f}", display=False)
glue("ms_batch", f"{1e3 * bm['float32']['batch_10000']['seconds_per_read']:.0f}", display=False)
glue("mb_nodes", f"{bm['float32']['resident_mib']:.0f}", display=False)
glue("worst_all", tex(q["all"]["sinc"]["local_peak"]["worst"]), display=False)
glue("iwa", f"{q['design_iwa_lod']}", display=False)
glue("ms_batch200", f"{1e3 * bm['float32']['batch_small']['seconds_per_read_median']:.0f}", display=False)
slower = bm["float32"]["single_steady_median_seconds"] / bm["yippy_blend"]["single_steady_median_seconds"]
glue("slower_than_blend", f"{slower:.0f}", display=False)

# Spacing rule for the starter plan and the starter's declared support widths.
B_PLAN = 1.0 * 1.0  # sigma_max times L
H_PLAN = 0.76 / (2 * B_PLAN)
DECLARED_L = (0.946, 0.982)  # declared illuminated widths along x and y, in D
rho_x, rho_y = (2 * width * H_PLAN for width in DECLARED_L)
glue("B_plan", f"{B_PLAN:.0f}", display=False)
glue("h_plan", f"{H_PLAN:.2f}", display=False)
glue("n_edge_x", f"{4.18 / H_PLAN:.0f}", display=False)
glue("rho_x", f"{rho_x:.3f}", display=False)
glue("rho_y", f"{rho_y:.3f}", display=False)
glue("L_x", f"{DECLARED_L[0]}", display=False)
glue("L_y", f"{DECLARED_L[1]}", display=False)
```

Image simulators, yield codes and exposure-time calculators need the
point spread function (PSF) of a planet wherever the planet happens to be,
but a coronagraph designer can deliver only a finite grid of propagated
images, one for each grid point (a **node**). Whatever fills the space
between nodes puts its error straight into planet photometry. How well can
the gaps be filled, and how would we know?

**yippy.sinc reads a planet's PSF at any position between the nodes of a
delivered grid.** In {glue:text}`n_reads` checks it reproduced a direct
propagation to within {glue}`worst_all` of the planet's brightest nearby pixel
(the local peak, defined under the figure below). That is a measured error,
not a bound, and every read says so. When the grid does not meet the
conditions the method needs, the read is refused with a stated reason.

The method is the one sampling theory makes exact. A signal whose fastest
oscillation is known (it is **band-limited**,
{doc}`foundations/sampling`) is fully determined by samples taken densely
enough, and each value between the samples is a sum of samples weighted by the
**sinc** function ({doc}`foundations/sinc-interpolation`). The yippy.sinc
library does this with a tapered, 32-sample version of the sinc, checks
before every use (**admission**) that the grid is dense enough, and refuses
when it is not. It is written in [JAX](https://github.com/jax-ml/jax) and
[Equinox](https://github.com/patrick-kidger/equinox). A node is one delivered
image, the thing the foundations pages call a sample.

The reader takes grids delivered for it: a regular grid of PSFs at a spacing
chosen from the optics, with independent checks and declarations
({doc}`practice/designers`). The starter delivery used on these pages is the
first real grid built this way, the public LUVOIR-A apodized pupil Lyot
coronagraph (APLC) propagated with physicaloptix on {glue:text}`n_nodes`
nodes. Existing YIP packages (yippy input products, the FITS and CSV files
that describe a coronagraph's off-axis PSFs) keep working with yippy's
**blend interpolation**, a weighted sum of the four nearest delivered images
each shifted to the query, and are not read by yippy.sinc.

```{figure} _static/validation/worst_check.png
:alt: Four panels around a check position at about (4, 5.4) lambda0/D in the public LUVOIR-A APLC. Direct propagation and the yippy.sinc read look identical. The yippy.sinc residual is blank at a scale of one part in a million. The blend residual shows a dipole of a few percent.
:width: 100%

Look at the two right panels, drawn on separate color scales: the yippy.sinc
residual is blank at the scale of a millionth, and the blend residual is a
dipole at the planet core. A planet between grid nodes of the public LUVOIR-A
APLC, propagated directly (left) and read from the delivered grid. The circle
marks the radius of 1 $\lambda_0/D$ around the planet that defines the local
peak, the brightest direct pixel inside it. Details on {doc}`validation`.
```

At this position the yippy.sinc read is within {glue}`sinc_here` of the local
peak, while blending neighboring images is off by {glue:text}`blend_here`
percent here and {glue:text}`blend_worst` percent at worst of the
{glue:text}`n_checks` starter checks.

:::{admonition} Who can use this today
:class: note

- **Monochromatic, off-axis products only.** Version 0.1 reads planet PSFs at
  one wavelength. Stellar and sky products stay with yippy, and band-mean
  products are declared but not yet demonstrated.
- **One real delivery exists.** The evidence is one design (the public
  LUVOIR-A APLC) read at {glue:text}`n_reads` positions, which include the
  {glue:text}`n_checks` independent starter checks, plus {glue:text}`n_mirror`
  mirror positions. The one-part-in-a-million figure is what those checks
  measured, not a bound: every read reports its error bound as unknown.
- **The checks cover one cell.** All of them lie in one cell between nodes
  near the design inner working angle ({glue:text}`iwa` $\lambda_0/D$, the
  separation at which the coronagraph is meant to start working). Measured per
  pixel against each pixel's brightest stored value, the worst error is about
  {glue}`pixel_all`.
- **A delivery costs propagations.** The starter is {glue:text}`n_nodes` PSFs
  (about {glue:text}`mb_nodes` MiB) plus {glue:text}`n_all_checks` checks, at
  a spacing set by the optics ({doc}`theory/band-limits`) rather than by
  convenience.
- **Reads cost more than yippy's blend.** A steady single read takes about
  {glue:text}`slower_than_blend` times as long as a blend read
  ({doc}`reference/performance`).
:::

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} You need PSFs at arbitrary planet positions
Read a delivered grid through yippy with `interpolation="sinc"`, or
directly with yippy.sinc. Start with {doc}`practice/first-read`, then
{doc}`practice/yippy`.
:::

:::{grid-item-card} You design coronagraphs and deliver grids
What spacing, which positions, which checks and which declarations a
readable delivery needs: {doc}`practice/designers`.
:::

:::{grid-item-card} You want to know why it works
Start with {doc}`foundations/sampling` and {doc}`foundations/sinc-interpolation`
(what perfect interpolation needs), then {doc}`theory/kernel` (what the 32-tap
version costs), {doc}`theory/band-limits` (why a coronagraph PSF is
band-limited) and {doc}`theory/quadrant-symmetry` (how symmetry halves the
grid twice).
:::

:::{grid-item-card} You want to know what it does not do
Version 0.1 does not certify accuracy: error bounds are reported as
unknown, and empirical checks are evidence, not bounds. See
{doc}`practice/admission` and the {doc}`roadmap`.
:::
::::

## What version 0.1 does

```{mermaid}
flowchart LR
    D["Designer grid<br/>PSFs, offsets, checks,<br/>declarations"] --> Y["yippy<br/>files, hashes, units"]
    Y --> A["admit_grid<br/>geometry, premises,<br/>sampling, symmetry"]
    A -->|admitted| I["Interpolator<br/>32 x 32 Kaiser-sinc reads"]
    A -->|refused| R["reasons"]
    I --> P["planet PSF<br/>at any served position"]
    I -->|outside| R
```

Admission asks whether the grid is dense enough (the oversampling factor
$\rho$ of {doc}`foundations/sampling` is at most 0.76), whether the symmetry
it declares was checked, whether each declared physical premise comes with
evidence, and whether the memory budget is met. A read is served only if its
$32\times32$ stencil of nodes exists, and it is refused otherwise. A
**Kaiser-sinc** read is the sinc of
{doc}`foundations/sinc-interpolation` multiplied by a smooth taper
({doc}`theory/kernel`).

- Reads off-axis planet intensity products from a complete grid of signed
  positions or from a positive quadrant with declared, independently
  checked mirror symmetry in both axes.
- Admits a grid only if $\rho\le0.76$ on every axis, meaning the grid spacing
  is at most 76 percent of the Nyquist spacing, the largest spacing at which
  sampling is still exact. At that limit the 32-tap kernel's worst error on
  any single cosine inside the band is measured at about {glue}`floor_tone`.
- Returns images in the delivered units with float64 arithmetic, under
  `eqx.filter_jit` and `jax.vmap`, at {glue:text}`ms_f32` ms per 128 by 128 PSF
  with float32 storage ({glue:text}`ms_f64` ms with float64 storage) in steady
  state, and {glue:text}`ms_batch200` ms per read in a batch of 200 positions,
  on a loaded laptop CPU.
- Refuses with named reasons whenever a declaration is missing, a check
  fails or a read falls outside the served region.

A second read profile trades accuracy for speed. It filters the grid once at
load and then reads from a $7\times7$ block of filtered images instead of the
$32\times32$ block of delivered ones, on a grid sampled at $\rho\le0.64$, and
it renders a scene of many sources in one pass
({doc}`practice/prefiltered-read`).

## How a grid is sized

The delivered spacing is not a convenience. One rule chains five quantities
from the optics to the grid, and every delivery in these pages follows it.

1. **Support width.** The illuminated pupil is $L$ telescope diameters
   $D$ wide. The planning value is $L=1$.
2. **Bandwidth.** As the planet moves, the brightness at one detector pixel
   cannot oscillate faster than $B=\sigma_{\max}L$ cycles per $\lambda_0/D$ of
   motion, which is {glue:text}`B_plan` for a monochromatic product with
   $\sigma_{\max}=1$ ({doc}`theory/band-limits` derives this from the pupil).
3. **Spacing.** Sampling is exact only if the oversampling factor
   $\rho=2Bh$ stays below 1, and the 32-tap kernel is accurate only up to
   $\rho=0.76$. Solving for the spacing gives $h=0.76/(2B)=$
   {glue:text}`h_plan` $\lambda_0/D$. The Nyquist spacing, at which $\rho=1$,
   is $1/(2B)$, here 0.5, so $\rho$ is also the ratio of the grid spacing to
   the Nyquist spacing.
4. **Node count.** Serving positions up to $R$ along an axis needs
   $\lceil R/h\rceil$ nodes plus 16 for the stencil: for $R=4.18$ along $x$,
   {glue:text}`n_edge_x` plus 16 is 27.
5. **Grid.** The nodes form a quadrant, $27\times31$ counting nodes along $x$
   and then along $y$.

```{code-cell} python
:tags: [remove-input]
from optixstuff.grids import Frame
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import yippy.sinc as sc
from yippy.sinc.geometry import source_grid
from yippy.sinc import viz

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
# The code orders a lattice shape as (n_y, n_x), so the 27 by 31 grid is (31, 27).
starter = source_grid(jnp.zeros(2), jnp.full(2, H_PLAN), (31, 27), Frame(500.0, 6.0, "circumscribed"))
fig, ax = plt.subplots(figsize=(10, 2.5), constrained_layout=True)
viz.plot_spacing_rule(lattice=starter, support_width=1.0, sigma_max=1.0, ax=ax)
plt.show()
```

The strip draws the five steps from left to right. In the first panel the
circle is the illuminated pupil and the arrow its width $L$. The second panel
is a sketch of the frequencies a pixel's brightness can contain as the planet
moves, which reach zero at $\pm B$.

In the third panel the gray dots are
nodes at the Nyquist spacing 0.5 (a spacing in $\lambda_0/D$, not the
Nyquist frequency), and the cyan dots are nodes at the spacing the kernel
allows, closer by the factor $\rho$. The fourth panel counts the nodes up to
the served edge in cyan and the 16 nodes of stencil margin in gray, and the
last shows the stored quadrant (cyan) and the box of positions whose mirrored
stencils are complete (gray). {doc}`theory/band-limits` derives the bandwidth
and {doc}`theory/quadrant-symmetry` builds the node count and the grid.

The plan used $L=1$, so $\rho={glue:text}`B_plan`\times2\times$
{glue:text}`h_plan` $=0.76$ on both axes. The starter's declared pupil widths
are smaller, {glue:text}`L_x` by {glue:text}`L_y` $D$ (from the delivery's
declaration, {doc}`practice/declarations`), which gives $\rho=2Bh=$
{glue:text}`rho_x` along $x$ and {glue:text}`rho_y` along $y$, both under the
limit.

## Install

```bash
pip install yippy
```

installs the library. Add the plotting module `yippy.sinc.viz` with
`pip install "yippy[viz]"`.

Sinc construction works in either JAX precision. Enable JAX 64-bit mode
before building a reader when you want binary64 read or scene arithmetic.

```{toctree}
:maxdepth: 1
:caption: Getting started
:hidden:

practice/first-read
```

```{toctree}
:maxdepth: 1
:caption: For users
:hidden:

practice/yippy
practice/admission
practice/prefiltered-read
```

```{toctree}
:maxdepth: 1
:caption: For designers
:hidden:

practice/designers
practice/declarations
practice/delivery-recipe
practice/recipe-frame-pixel
practice/recipe-band-symmetry
practice/recipe-check-send
validation
```

```{toctree}
:maxdepth: 1
:caption: Understanding
:hidden:

foundations/sampling
foundations/sinc-interpolation
theory/kernel
theory/toy-coronagraph
theory/two-dimensional-read
theory/band-limits
theory/quadrant-symmetry
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

reference/numerical-contract
reference/stencil-floor
reference/performance
reference/glossary
viz
roadmap

```
