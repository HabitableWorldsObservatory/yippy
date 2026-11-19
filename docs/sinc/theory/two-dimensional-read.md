---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  render_markdown_format: myst
---

# What does reading a two-dimensional grid cost?

A planet's image depends on two source coordinates, so the grid of images has
two axes and the kernel of the {doc}`previous pages <kernel>` has to be used on
both. This page counts what one read then adds up, and watches the sum settle
onto its final value for the probe pixel of the {doc}`toy coronagraph
<toy-coronagraph>`.

:::{admonition} What perfect interpolation needs
:class: important

1. **A band limit.** The signal has no cosine above a known bandwidth
   $f_{\max}$. *({doc}`band-limits`)*
2. **Dense enough samples.** $\rho=2f_{\max}<1$. *({doc}`../foundations/sampling`)*
3. **Weights a program can afford, on two axes.** The Kaiser-sinc kernel,
   applied once along each axis. **This page.**
:::

```{code-cell} python
:tags: [remove-cell]
import tempfile
from pathlib import Path

import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, Video
from myst_nb import glue

import eyepiece as ep
import toy_coronagraph as tc
from yippy.sinc import viz
from yippy.sinc.kernel import KernelProfile, stencil

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette
FAINT = ep.style.neutral(0.25)
SCENERY = ep.style.neutral(0.45)
POSITIVE = ep.style.role("positive")
NEGATIVE = ep.style.role("negative")

_clips = iter(range(100))


def clip(anim):
    """Embed an animation as an mp4 with the browser's own player controls."""
    path = Path(tempfile.gettempdir()) / f"yippy-sinc-docs-read2d-{next(_clips)}.mp4"
    anim.video(path, dpi=200)
    return Video(str(path), embed=True, html_attributes="controls loop muted playsinline")


def num(name, text):
    """Glue a plain string."""
    glue(name, text, display=False)


def tex(value, digits=1):
    """Format a number as a LaTeX power of ten."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return Markdown(rf"${mantissa}\times10^{{{int(exponent)}}}$")


toy = tc.ToyCoronagraph()
x_pix = np.asarray(toy.detector_x)
col = int(np.argmin(np.abs(x_pix - 3.1)))
row = int(np.argmin(np.abs(x_pix - 0.6)))

# A square grid of nodes 0.38 apart whose first node sits 16 spacings below zero,
# so every query near the origin has its whole stencil.
SPACING = 0.38
grid = ((-16 * SPACING, -16 * SPACING), (SPACING, SPACING), (40, 40))
QUERY = np.array([1.3, 0.7])


def one_read(query):
    """Taps, weights and node values of one read of the probe pixel."""
    tx, wx = stencil(jnp.asarray([query[0]]), grid[0][0], grid[1][0])
    ty, wy = stencil(jnp.asarray([query[1]]), grid[0][1], grid[1][1])
    node_x = grid[0][0] + np.asarray(tx[0]) * grid[1][0]
    node_y = grid[0][1] + np.asarray(ty[0]) * grid[1][1]
    tap_xy = np.stack(np.meshgrid(node_x, node_y), axis=-1).reshape(-1, 2)
    weight = np.outer(np.asarray(wy[0]), np.asarray(wx[0])).ravel()
    value = np.asarray(toy.psf(tap_xy))[:, row, col]
    return node_x, node_y, np.asarray(wx[0]), np.asarray(wy[0]), weight, value


node_x, node_y, wx, wy, tap_weight, tap_value = one_read(QUERY)
```

## How is a two-dimensional weight built from one-dimensional ones?

The kernel gives one weight for each of the 32 nodes along an axis. A read at the
query $(s_x,s_y)$ weights the node in column $i$ and row $j$ by the product of the
two one-dimensional weights, $w(t_x)\,w(t_y)$, the weight of its column times the
weight of its row. A weight built this way is called **separable**. The
32 columns and 32 rows make $32\times32=1024$ taps. The figure shows the column
weights across the top, the row weights down the right, and their products in the
centre as markers whose area is the size of the weight, for the query
$(1.3,\,0.7)\,\lambda_0/D$ on a grid of nodes $0.38\,\lambda_0/D$ apart.

```{code-cell} python
:tags: [hide-input]
fig = plt.figure(figsize=(8, 7.4), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[5, 1.4], height_ratios=[1.4, 5])
top = fig.add_subplot(gs[0, 0])
main = fig.add_subplot(gs[1, 0], sharex=top)
side = fig.add_subplot(gs[1, 1], sharey=main)

viz.plot_stencil_2d(QUERY, grid, profile=KernelProfile(), order="weight",
                    n_taps=1024, max_area=450.0, ax=main)
span_x = (node_x[0] - SPACING, node_x[-1] + SPACING)
span_y = (node_y[0] - SPACING, node_y[-1] + SPACING)
main.set_xlim(*span_x)
main.set_ylim(*span_y)
main.set_xlabel(r"node position $s_x$ ($\lambda_0/D$)")
main.set_ylabel(r"node position $s_y$ ($\lambda_0/D$)")
for ax, nodes, w, horizontal in ((top, node_x, wx, False), (side, node_y, wy, True)):
    colors = [POSITIVE if v > 0 else NEGATIVE for v in w]
    if horizontal:
        ax.hlines(nodes, 0, w, colors=colors, lw=1.6)
        ax.plot(w, nodes, "o", color=SCENERY, ms=3)
        ax.axvline(0, color=FAINT, lw=1)
        ax.set_xlabel(r"$w(t_y)$")
        ax.tick_params(labelleft=False)
    else:
        ax.vlines(nodes, 0, w, colors=colors, lw=1.6)
        ax.plot(nodes, w, "o", color=SCENERY, ms=3)
        ax.axhline(0, color=FAINT, lw=1)
        ax.set_ylabel(r"$w(t_x)$")
        ax.tick_params(labelbottom=False)
plt.show()
num("r_n_taps", f"{wx.size * wy.size}")
num("r_wmax", f"{np.abs(tap_weight).max():.2f}")
glue("r_wmin", tex(np.abs(tap_weight).min()), display=False)
num("r_wsum", f"{tap_weight.sum():.6f}")
```

Look at how the centre is built from the two edges. The strongest column weight
and the strongest row weight multiply to the largest tap, {glue:text}`r_wmax`,
and the weights alternate in sign along both edges, so the centre alternates in
sign along both axes. The smallest of the {glue:text}`r_n_taps` weights is
{glue}`r_wmin`, and all of them add to {glue:text}`r_wsum`.

The cost follows from the count. A read adds {glue:text}`r_n_taps` products for each
detector pixel, however large the grid is, because only the taps of the stencil
are touched. An image of {glue:text}`r_npix` by {glue:text}`r_npix` pixels at one
planet position therefore adds {glue:text}`r_image_cost` products.

```{code-cell} python
:tags: [remove-cell]
num("r_npix", f"{toy.n_pix}")
num("r_image_cost", f"{toy.n_pix**2 * wx.size * wy.size:,}")
```

## Why are many taps needed to settle on a faint value?

A read is a sum of 1024 products of a weight and a node value. The figure shows
the node values the sum multiplies, for the probe pixel, on a logarithmic scale.
The planet at $(1.3,\,0.7)$ is behind the occulter, so its read is faint, while
the brightest node values in the block are far brighter.

```{code-cell} python
:tags: [hide-input]
values_map = tap_value.reshape(wy.size, wx.size)
fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
image = ax.imshow(np.log10(np.maximum(values_map, 1e-12)), origin="lower", cmap="magma",
                  extent=(node_x[0] - SPACING / 2, node_x[-1] + SPACING / 2,
                          node_y[0] - SPACING / 2, node_y[-1] + SPACING / 2),
                  interpolation="nearest")
ax.add_patch(plt.Circle((0, 0), toy.occulter_lod, fill=False, color=SCENERY, lw=1, ls="--"))
ax.plot(*QUERY, "+", color=pal.pink, ms=11, mew=1.8)
ax.set_xlabel(r"node position $s_x$ ($\lambda_0/D$)")
ax.set_ylabel(r"node position $s_y$ ($\lambda_0/D$)")
fig.colorbar(image, ax=ax, label="probe pixel intensity at the node")
image.colorbar.set_ticks([-9, -6, -3, 0])
image.colorbar.set_ticklabels([r"$10^{-9}$", r"$10^{-6}$", r"$10^{-3}$", r"$10^{0}$"])
plt.show()

read = float(np.sum(tap_weight * tap_value))
glue("r_read", tex(read), display=False)
num("r_vmax", f"{tap_value.max():.2f}")
num("r_ratio", f"{tap_value.max() / read:.0f}")
```

The pink cross is the query, inside the dashed outline of the occulter. The read
of the probe pixel there is {glue:text}`r_read`, the node values reach
{glue:text}`r_vmax`, and the ratio is {glue:text}`r_ratio`. The sum has to cancel
nearly everything it adds.

```{admonition} Predict before reading on
:class: tip
The taps are added in order of decreasing weight, and the weights alternate in
sign. After how many taps does the partial sum stay within one part in a
thousand of the final read: about 30, about 300 or about 1000?
```

```{code-cell} python
:tags: [hide-input]
# The partial sum after the k heaviest taps, for this faint read.
by_size = np.argsort(-np.abs(tap_weight), kind="stable")
partial = np.cumsum((tap_weight * tap_value)[by_size])
final = partial[-1]
counts = np.unique(np.round(np.geomspace(1, 1024, 36)).astype(int))
frames = np.concatenate([[0, 0], counts, np.full(6, 1024)])

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1, 1.1]})
stencil_view = viz.plot_stencil_2d(QUERY, grid, profile=KernelProfile(), order="weight",
                                   n_taps=0, max_area=450.0, ax=left)
left.set_xlim(-5.5, 8.3)
left.set_ylim(-5.5, 8.3)
conv = ep.plot.convergence((tap_weight * tap_value)[by_size], running="sum", refs=final,
                           ref_labels=["read value"], show_samples=False, ax=right)
right.set_xscale("log")
right.set_xlim(1, 1024 * 3)
right.set_xlabel(r"taps counted $k$, largest weight first")
right.set_ylabel("partial sum (probe pixel)")
count_text = right.text(0.03, 0.93, "", transform=right.transAxes, va="top")
left.set_title("1024 taps, area is weight", fontsize=10)


def draw(fig, k):
    k = int(k)
    stencil_view.update(n_taps=k)
    conv.update(max(k, 1) if k else 0)
    ratio = rf", sum / read = {partial[k - 1] / final:.4f}" if k else ""
    count_text.set_text(rf"$k$ = {k}" + ratio)


anim = ep.render.animate(fig, draw, frames, fps=10)
clip(anim)
```

In the clip, taps are counted from the largest weight outward. On the left each
counted tap is a marker at its node, with area for its weight and colour for its
sign, and the taps not yet counted are faint dots. On the right the running sum
of weight times node value approaches the read value.

The still below runs the same count for two reads and shows the distance of the
partial sum from the final value, as a fraction of it. The faint read is the
one in the clip. The bright read has the planet on the probe pixel, at
$(3.3,\,0.5)$, the position of the pixel's peak on the previous page.

```{code-cell} python
:tags: [hide-input]
def settle(query):
    """Relative distance of the partial sum from the read, and the k that stays within 1e-3."""
    *_, weight, value = one_read(query)
    order = np.argsort(-np.abs(weight), kind="stable")
    running = np.cumsum((weight * value)[order])
    distance = np.abs(running - running[-1]) / abs(running[-1])
    return distance, int(np.nonzero(distance > 1e-3)[0].max()) + 2, running[-1]


faint_distance, faint_k, faint_read = settle(QUERY)
bright_query = np.array([3.3, 0.5])
bright_distance, bright_k, bright_read = settle(bright_query)
ks = np.arange(1, 1025)

fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
ax.loglog(ks[:-1], np.maximum(faint_distance[:-1], 1e-16), color=pal.cyan, lw=1.8)
ax.loglog(ks[:-1], np.maximum(bright_distance[:-1], 1e-16), color=pal.yellow, lw=1.8)
ax.axhline(1e-3, color=SCENERY, lw=1, ls="--")
ax.axvline(faint_k, color=pal.cyan, lw=1, ls=":")
ax.axvline(bright_k, color=pal.yellow, lw=1, ls=":")
ax.set_ylim(1e-9, 1e3)
ax.text(1.3, 2e-3, "one part in a thousand", color=SCENERY, fontsize=9)
ax.text(1.3, 3e-7, rf"faint read, query $(1.3,\,0.7)$: settles at $k={faint_k}$",
        color=pal.cyan, fontsize=10)
ax.text(1.3, 2e-8, rf"bright read, query $(3.3,\,0.5)$: settles at $k={bright_k}$",
        color=pal.yellow, fontsize=10)
ax.set_xlabel(r"taps counted $k$, largest weight first")
ax.set_ylabel("distance of the sum from the read\n(fraction of the read)")
plt.show()
num("r_faint_k", f"{faint_k}")
num("r_bright_k", f"{bright_k}")
num("r_faint_first", f"{partial[29] / final:.1f}")
num("r_overshoot", f"{partial.max() / final:.0f}")
num("r_bright_read", f"{bright_read:.2f}")
```

```{dropdown} Answer
About 300: for the faint read the sum stays within one part in a thousand
of the read from $k=$ {glue:text}`r_faint_k` on. It is nowhere near that early:
after 30 taps it is {glue:text}`r_faint_first` times the read, and the largest
partial sum is {glue:text}`r_overshoot` times the read.
```

Look at the two vertical dotted lines. The faint read needs {glue:text}`r_faint_k`
taps to settle and the bright read, whose value is {glue:text}`r_bright_read`,
needs {glue:text}`r_bright_k`. Two things set the count: the weights alternate in
sign, so large partial sums cancel, and a faint read leaves little room for the
cancellation to be imperfect. A fraction of a faint value is a small absolute
error, so the accuracy of the previous page, which is a fraction of the signal's
amplitude and not of the local value, is a weaker promise for a read that is much
fainter than its neighbours. {doc}`band-limits` measures this on the probe pixel.

## Try it

1. Predict, then change `QUERY` in the setup to $(1.3,\,0.0)$, a planet exactly on
   a row of nodes. How many taps carry a weight above $10^{-10}$ of the largest?
   ```{dropdown} Hint
   The row weight $w(t_y)$ is zero at every integer but one, so only one row of
   32 taps carries the read, and the other rows weigh about $10^{-16}$.
   ```
2. How many products does the probe pixel's whole curve of 961 positions, as in the
   clip of the {doc}`toy page <toy-coronagraph>`, add to read from the grid?
   ```{dropdown} Hint
   $961\times1024$, because every position adds its own 32 by 32 block.
   ```
3. Use `settle` from the last figure's cell for a planet on the far side of the
   occulter from the probe pixel, $(-3.3,\,0.5)$. Predict first: does the read
   settle sooner or later than the bright one?
   ```{dropdown} Hint
   The pixel is far from that planet's core, so the read is faint, and a faint
   read settles later, as in the figure.
   ```

## What this page established

:::{admonition} What perfect interpolation needs
:class: important

1. **A band limit.** The signal has no cosine above a known bandwidth
   $f_{\max}$.
2. **Dense enough samples.** $\rho=2f_{\max}<1$.
3. **Weights a program can afford, on two axes.** Each tap weighs the product of
   a column weight and a row weight, a read adds {glue:text}`r_n_taps` products per
   pixel, and a faint read needs a few hundred of them before it settles.
:::

The two-dimensional read is the kernel applied along each axis, at a cost that does
not depend on the grid. The next page returns to the first requirement and asks what
bandwidth a coronagraph pixel has.

Next: {doc}`band-limits`.
