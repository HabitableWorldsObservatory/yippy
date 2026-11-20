---
file_format: mystnb
kernelspec:
  name: python3
---

# Visualization gallery

The `yippy.sinc.viz` module draws yippy.sinc's own objects (kernels, stencils, lattices,
reads, receipts) on top of [eyepiece](https://eyepiece.readthedocs.io). It
is an optional extra, `pip install "yippy[viz]"`, and importing yippy.sinc
never imports it. Every function takes either the yippy.sinc object or plain
arrays, draws into an `ax` you pass or a new figure, and returns an eyepiece
`PlotResult`. Every function except `plot_kernel`, `plot_spacing_rule` and
`plot_check_residuals` also provides an `update` method that redraws in place
for animations. The
figures on the theory and practice pages are built from these functions.

:::{admonition} What the gallery cells need
:class: note

The cells for `plot_read`, `plot_residual`, `plot_check_residuals` and
`plot_band_spectrum` run on the toy coronagraph in `docs/toy_coronagraph.py`
of the repository (a Lyot coronagraph computed on the page, described on
{doc}`theory/band-limits`). That file is documentation scaffolding and not part of the
package: it supplies images and a delivery to draw. Every other cell needs
only yippy.sinc and eyepiece.
:::

```{code-cell} python
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import yippy.sinc as sc
from yippy.sinc.geometry import source_grid
from yippy.sinc import viz
from yippy.sinc.kernel import KernelProfile
import toy_coronagraph as tc

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
```

## `plot_kernel`

The Kaiser-sinc weights ({doc}`foundations/sinc-interpolation`, {doc}`theory/kernel`)
against offset from the query, with the ideal sinc behind them and the support
edges at $\pm16$ samples. Look at the tails: the sinc keeps oscillating past the
edges, and the tapered kernel decays smoothly to zero at them.

```{code-cell} python
result = viz.plot_kernel()
plt.show()
```

## `plot_stencil`

One axis of a read: the stencil taps of the profile you pass (or of a reader) and their weights at a query
({doc}`foundations/sinc-interpolation`), here on a grid of 30 stored nodes. Look at
the hollow stems: they are taps at node indices below zero, before the first
stored node, and a read with such taps is refused unless the grid is a quadrant
that reads them by mirror symmetry.

```{code-cell} python
fig, ax = plt.subplots(figsize=(7, 2.4), constrained_layout=True)
result = viz.plot_stencil(3.3, profile=KernelProfile(), origin=0.0, spacing=1.0,
                         n_nodes=30, ax=ax)
plt.show()
```

## `plot_stencil_2d`

The taps of one read over the node lattice (the 32 by 32 block of the raw
stencil in two dimensions; pass a reader to draw the stencil it reads with), marker area proportional to the weight and color for
its sign. Look at the few large markers near the query: most of the taps
are too small to see. Taps are counted largest first with `order="weight"`.
`n_taps` sets how many are counted, and `fold` (quadrant lattices only, 0 to 1)
slides each tap that lies at a negative node onto the stored node it reads
through the mirror symmetry of {doc}`theory/quadrant-symmetry`. This cell needs
no delivery, only a lattice.

```{code-cell} python
fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
result = viz.plot_stencil_2d((1.0, 2.0), ((0.0, 0.0), (0.38, 0.38), (31, 31)),
                             profile=KernelProfile(), quadrant=True,
                             order="weight", n_taps=200,
                             fold=1.0, max_area=600.0, ax=ax)
ax.set_xlim(-6, 8)
ax.set_ylim(-4, 10)
plt.show()
```

## `plot_coverage`

Stored nodes, mirrored nodes of a quadrant grid, the region where complete
stencils exist (the served region of the {doc}`reference/glossary`), a domain
and one read's stencil rectangle. Look at where the stencil rectangle sits
against the region: a read is served only if the rectangle lies inside it.

```{code-cell} python
from optixstuff.grids import Frame
lattice = source_grid(jnp.zeros(2), jnp.full(2, 0.38), (31, 31), Frame(500.0, 6.0, "circumscribed"))
domain = sc.Domain("box", "closed", jnp.array([[-5.0, -5.0], [5.0, 5.0]]), jnp.zeros(2))
fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
result = viz.plot_coverage(lattice, domain=domain, quadrant=True, query=(1.0, 2.0), ax=ax)
plt.show()
```

## `plot_spacing_rule`

The chain from support width to the delivered grid: bandwidth, largest
admissible spacing, node count and the quadrant lattice, drawn panel by panel on
the {doc}`index` page and explained on {doc}`theory/band-limits` (bandwidth) and
{doc}`theory/quadrant-symmetry` (node count and grid). Pass physical inputs,
or a `Products` and a `optixstuff.grids.PixelGrid` to draw a delivery. This cell needs neither.

```{code-cell} python
fig, ax = plt.subplots(figsize=(10, 2.6), constrained_layout=True)
result = viz.plot_spacing_rule(support_width=1.0, radius=(4.18, 5.70), ax=ax)
plt.show()
```

## `plot_read` and `plot_residual`

A read on a logarithmic scale in $\lambda_0/D$, and its difference from a
reference image relative to the local peak (the brightest reference pixel
within 1 $\lambda_0/D$ of the source, see the {doc}`reference/glossary`), with that
radius drawn around the source. Look at the right panel: the residual is a
small fraction of the local peak.

```{code-cell} python
toy = tc.ToyCoronagraph()
delivery = tc.quadrant_delivery()
reader = tc.reader_for(delivery)
query = jnp.array([3.2, 1.1])
read = reader.read(query, jnp.int32(0))
truth = toy.psf(query)
pixels = delivery["products"].pixels
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
viz.plot_read(read, pixels=pixels, ax=a)
viz.plot_residual(read, truth, source_xy_lod=query, pixels=pixels, ax=b)
for ax in (a, b):
    ax.set_xlim(-1, 7)
    ax.set_ylim(-3, 5)
plt.show()
```

## `plot_check_residuals`

The empirical evidence in an admission receipt: interior checks, mirror
checks and axis invariance, with exact zeros gathered at the left edge as
triangles. The dashed lines mark the target of $10^{-5}$ for the interior checks
and the symmetry tolerance of $10^{-6}$ for the other two
({doc}`validation`, {doc}`reference/numerical-contract`).

```{code-cell} python
admission = sc.admit_grid(**delivery)
result = viz.plot_check_residuals(admission.receipt)
plt.show()
```

## `plot_band_spectrum`

The amplitude spectrum of a function sampled along source position, with
the pupil band shaded. The frequency axis is in cycles per $\lambda_0/D$ of
source motion, and the call below passes the sample step 0.05 and a bandwidth
of 1 in those units ({doc}`foundations/sampling`). Look for the spectrum falling
away to the right of the shaded band.

```{code-cell} python
s = np.arange(-512, 512) * 0.05
images = toy.psf(np.stack([s, np.full_like(s, 0.5)], axis=1))
trace = np.asarray(images[:, 50, 60])
result = viz.plot_band_spectrum(trace, 0.05, bandwidth=1.0)
plt.show()
```

Next: {doc}`roadmap`.
