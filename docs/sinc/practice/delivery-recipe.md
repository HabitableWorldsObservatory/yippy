---
file_format: mystnb
kernelspec:
  name: python3
---

# How do you produce a delivery that admits the first time?

A designer who follows the physics checklist can still send files that are
refused for a bookkeeping reason, or worse, files that admit and are wrong.
Either outcome costs a propagation campaign, because the images cannot be
patched afterwards. The physics is on {doc}`designers` and the file contract
is on yippy's [schema page][schema]. This recipe is the order of operations
around them, in six steps spread over four pages.

:::{admonition} The question of these four pages
:class: important

In what order do you plan the positions, fix the frame and units, handle the
band and the symmetry, then check and send the files, so that the delivery
admits the first time and is right? This page plans the positions (step 1).
{doc}`recipe-frame-pixel` fixes the frame and the pixel measure (steps 2 and
3), {doc}`recipe-band-symmetry` handles the filter and the symmetry argument
(steps 4 and 5), and {doc}`recipe-check-send` checks the files and sends them
(step 6).
:::

```{code-cell} python
:tags: [remove-cell]
import csv
import sys

sys.path.insert(0, "../scripts")
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from myst_nb import glue
from validation_tables import load

import yippy.sinc as sc
from yippy.sinc import viz

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
```

## What are the two examples that run through the recipe?

The first is the public starter request, which the numbers on
{doc}`../validation` and {doc}`yippy` describe: a quadrant grid for the public
LUVOIR-A APLC. The cell reads its sizes from the committed validation data
and not from this page.

```{code-cell} python
v = load("validation.json")
declared = load("floor_diagnosis.json")["band"]["declared_half_band"]
print(f"starter nodes {v['training_nodes']} = 27 * 31, spacing 0.38 lambda0/D")
print(f"interior checks {v['independent_checks']}, mirror checks {v['mirror_checks']}")
print(f"declared illuminated widths {declared['x']} by {declared['y']} D")
glue("n_nodes", str(v["training_nodes"]), display=False)
glue("n_checks", str(v["independent_checks"]), display=False)
glue("n_mirror", str(v["mirror_checks"]), display=False)
```

So the starter is {glue:text}`n_nodes` nodes, {glue:text}`n_checks` interior
checks and {glue:text}`n_mirror` mirror checks, with a 128 by 128 detector
frame of 0.25 $\lambda_0/D$ pixels ({doc}`yippy` describes the files). The
second example is the toy coronagraph of the theory pages, which stands in
wherever a step needs a measurement the starter cannot give, because the
starter is not on disk. The toy has a 96 by 96 pixel frame of the same
0.25 $\lambda_0/D$ pixels, so a number measured on the toy depends on 96 where
the starter has 128, and the box {ref}`toy-in-numbers` lists the toy's other
parameters. The toy images are also point samples of the intensity, which
exercises admission and not the pixel-integral contract that step 3 recommends.

(recipe-positions)=
## Step 1: Which positions do you propagate?

The goal of this step is the list of source positions to propagate, which the
planner writes for you. Run it with the reference wavelength and the shortest
contributing wavelength, both in meters, the illuminated entrance width in
units of $D$, and the served limits in $\lambda_0/D$. Two limits give a box,
one limit gives a circle. The call below uses the starter's values (1 micron
for both wavelengths) and the cell layout of checks, which places the
{glue:text}`n_checks` interior checks in the cell next to the outermost
node instead of spreading them over the radius, and it reproduces the
starter. The starter is not on disk here, so the command is shown and not run.

```bash
yippy-sinc plan --lambda0 1e-6 --lambda-min 1e-6 --support 1 \
    --limit 4.18 5.7 --check-layout cell --out coords
# spacing 0.38 lambda0/D, rho 0.7600, 27 x 31 nodes (837 training),
# 16 checks, 20 mirror checks
```

Three rules sit behind the output, and each can be checked by hand. The
spacing is the largest $h$ with $\rho=2(\lambda_0/\lambda_{\min})Lh\le0.76$,
rounded down to six decimals, using the larger support width. For $L=1$ at
$\lambda_0$ that is 0.38, as {doc}`designers` derived. The nodes per axis are
$\lceil R/h\rceil+16$ because a read near the served limit reaches 16 nodes
beyond it. The cell applies both rules to the starter's limits.

```{code-cell} python
spacing = 0.76 / (2 * 1.0 * 1.0)  # L = 1, shortest wavelength = reference
limits = np.array([4.18, 5.7])
cells = np.ceil(limits / spacing).astype(int)
nodes = cells + 16
print(f"spacing {spacing}; ceil(R / h) = {cells}; nodes per axis = {nodes}; "
      f"total {nodes[0] * nodes[1]}")
assert nodes[0] * nodes[1] == v["training_nodes"]
```

The planner also writes the check list and the mirror-check list, and the
lists are used unchanged. Two further choices need a reason. The
{glue:text}`n_checks` interior checks are a default count, which `--checks`
changes. Sixteen is enough to see a spread of residuals and cheap to
propagate, and the same `--seed` always gives the same list. Each check is
off every node, so that it is a true between-node read.

The mirror checks are the images propagated directly at negative positions.
Each is a counterpart of a stored base node, and the figure shows where the
eight bases sit on the starter lattice. Look at the labeled cyan nodes: one
is near the origin, where the occulter edge sits, two are on the axes, three
are beside or at the corner of the served cell, and two are the outermost
stored column and row.

```{code-cell} python
:tags: [hide-input]
jx, jy = cells - 1
nx, ny = nodes
bases = {  # label: (node, label offset in points, horizontal alignment)
    r"$(1, 1)$": ((1, 1), (6, 6), "left"),
    r"$(1, j_y)$": ((1, jy), (6, -16), "left"),
    r"$(j_x, 1)$": ((jx, 1), (-8, 6), "right"),
    r"$(j_x, j_y)$": ((jx, jy), (6, 6), "left"),
    r"$(0, j_y)$": ((0, jy), (-8, 8), "right"),
    r"$(j_x, 0)$": ((jx, 0), (8, -12), "left"),
    r"$(n_x-1, 1)$": ((nx - 1, 1), (-6, 8), "right"),
    r"$(3, n_y-1)$": ((3, ny - 1), (8, -3), "left"),
}
ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
ax.scatter(ix * spacing, iy * spacing, s=6, color="0.5")
for label, ((i, j), offset, align) in bases.items():
    ax.scatter(i * spacing, j * spacing, s=60, color="cyan", zorder=3)
    ax.annotate(label, (i * spacing, j * spacing), xytext=offset, textcoords="offset points",
                fontsize=10, ha=align, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc=fig.get_facecolor(), ec="none"))
ax.add_patch(plt.Rectangle((0, 0), *(limits), fill=False, linestyle="--", color="white"))
ax.set_xlabel(r"$x$ [$\lambda_0/D$]")
ax.set_ylabel(r"$y$ [$\lambda_0/D$]")
ax.set_aspect("equal")
plt.show()
mirror_checks = 3 * 6 + 1 * 2
print(f"6 general bases * 3 + 2 axis bases * 1 = {mirror_checks}")
assert mirror_checks == v["mirror_checks"]
assert [list(b) for b in v["axis_base_indices"]] == [[0, jy], [jx, 0]]
```

Here $j_x=\lceil R_x/h\rceil-1$ and $j_y=\lceil R_y/h\rceil-1$, the last node inside
the served limit. A base with both indices nonzero has three negative
counterparts (flip $x$, flip $y$, both) and a base on an axis has one, so the
six general bases and two axis bases give $6\times3+2\times1=20$ mirror
checks, which is the {glue:text}`n_mirror` of the starter. The dashed
rectangle is the served region.

The two outer-edge bases, the last two in the label list, matter because a read
at a negative position near the served edge uses the mirror images of the
outermost stored nodes. A symmetry that fails only in the far wing, where the
stencil of an edge read lives, would be invisible to checks near the origin.
The validator adds a note, without failing, when no mirror base reaches the
largest stored index on an axis.

## What this page established

The positions to propagate are the lattice of $\lceil R/h\rceil+16$ nodes per
axis at the spacing from the 0.76 rule, the interior checks between nodes, and
the mirror checks at the eight bases, all written by the planner and used
unchanged. The next page fixes how each of those images is framed and what its
pixels hold.

Next: {doc}`recipe-frame-pixel`.

[schema]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html
