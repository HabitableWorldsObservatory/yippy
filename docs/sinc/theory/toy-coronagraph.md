---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  render_markdown_format: myst
---

# What does one pixel of a coronagraph do as a planet moves?

A grid of precomputed planet images is useful only if each pixel changes in a
way that the grid can follow. Before counting what a read of such a grid costs
({doc}`two-dimensional-read`) and asking how fast a pixel can change
({doc}`band-limits`), this page builds the small coronagraph that those pages
use and watches one pixel of its detector while a planet crosses the field.

:::{admonition} What perfect interpolation needs
:class: important

1. **A band limit.** The signal has no cosine above a known bandwidth
   $f_{\max}$. **This page builds the signal, a pixel's brightness as a planet
   moves; {doc}`band-limits` finds its bandwidth.**
2. **Dense enough samples.** $\rho=2f_{\max}<1$. *({doc}`../foundations/sampling`)*
3. **Weights a program can afford.** The Kaiser-sinc kernel. *({doc}`kernel`)*
:::

```{code-cell} python
:tags: [remove-cell]
import tempfile
from pathlib import Path

import hwostyle
import jax
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Video
from myst_nb import glue

import eyepiece as ep
from yippy.sinc import viz
from toy_coronagraph import ToyCoronagraph

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette
FAINT = ep.style.neutral(0.25)
SCENERY = ep.style.neutral(0.45)

_clips = iter(range(100))


def clip(anim):
    """Embed an animation as an mp4 with the browser's own player controls."""
    path = Path(tempfile.gettempdir()) / f"yippy-sinc-docs-toy-{next(_clips)}.mp4"
    anim.video(path, dpi=200)
    return Video(str(path), embed=True, html_attributes="controls loop muted playsinline")


def num(name, text):
    """Glue a plain string."""
    glue(name, text, display=False)


toy = ToyCoronagraph()
x_pix = np.asarray(toy.detector_x)
col = int(np.argmin(np.abs(x_pix - 3.1)))
row = int(np.argmin(np.abs(x_pix - 0.6)))
PATH_Y = 0.5
num("t_occ", f"{toy.occulter_lod:g}")
num("t_lyot", f"{toy.lyot:g}")
num("t_npix", f"{toy.n_pix}")
num("t_pitch", f"{toy.pixscale_lod:g}")
num("t_path_y", f"{PATH_Y:g}")
num("t_probe_x", f"{x_pix[col]:.3f}")
num("t_probe_y", f"{x_pix[row]:.3f}")
```

## What is the toy coronagraph?

A coronagraph blocks the light of a star so that a faint planet beside it can be
seen. The toy is a Lyot coronagraph reduced to four elements, each drawn below
from the arrays that this page computes. Light from a distant planet enters
through a circular entrance pupil of diameter $D$ (panel 1). A disk in the first
focal plane, the occulter, blocks the middle of the image (panel 2). A second
mask in the pupil plane, the Lyot stop, removes the light that diffracted around
the occulter's edge (panel 3). The detector records the result (panel 4).

```{code-cell} python
:tags: [hide-input]
uu, vv = toy.pupil_coords
uu, vv = np.asarray(uu), np.asarray(vv)
stop = (uu**2 + vv**2 <= (toy.lyot / 2) ** 2).astype(float)
shown = 3.8
fig, axes = plt.subplots(1, 4, figsize=(10, 3.9), constrained_layout=True)
open_color = ep.style.neutral(0.75)

for ax, mask, title in ((axes[0], np.asarray(toy.pupil), "1 entrance pupil\ndiameter $D$"),
                        (axes[2], stop, rf"3 Lyot stop" "\n" rf"diameter ${toy.lyot:g}\,D$")):
    ax.imshow(mask, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5), cmap="gray", vmin=0,
              vmax=1.4, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"pupil coordinate ($D$)")
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([])

axes[1].add_patch(plt.Circle((0, 0), toy.occulter_lod, color=SCENERY))
axes[1].set_xlim(-shown, shown)
axes[1].set_ylim(-shown, shown)
axes[1].set_aspect("equal")
axes[1].set_title(rf"2 occulter" "\n" rf"radius ${toy.occulter_lod:g}\,\lambda_0/D$", fontsize=9)
axes[1].set_xlabel(r"focal-plane position ($\lambda_0/D$)")
axes[1].set_yticks([])

path_x = np.linspace(-6, 6, 100)
first = np.asarray(toy.psf(np.array([[-4.5, PATH_Y]])))[0]
viz.plot_read(first, pixscale_lod=toy.pixscale_lod, ax=axes[3], vmin=1e-9, vmax=0.5,
              colorbar=False)
axes[3].add_patch(plt.Circle((0, 0), toy.occulter_lod, fill=False, color=SCENERY, lw=1, ls="--"))
axes[3].plot(path_x, np.full_like(path_x, PATH_Y), color=pal.pink, lw=1, alpha=0.7)
axes[3].plot(-4.5, PATH_Y, "+", color=pal.pink, ms=9, mew=1.5)
axes[3].plot(x_pix[col], x_pix[row], "s", mfc="none", mec=pal.yellow, ms=7, mew=1.5)
axes[3].set_xlim(-8, 8)
axes[3].set_ylim(-8, 8)
axes[3].set_title(rf"4 detector: ${toy.n_pix}\times{toy.n_pix}$ pixels," "\n"
                  rf"${toy.pixscale_lod:g}\,\lambda_0/D$ each", fontsize=9)
axes[3].set_yticks([])
plt.show()
```

Look at panel 4. The image is the planet's **point spread function** (PSF), the
pattern a point source makes on the detector, and it depends on where the planet
is. Here the planet (pink cross) is at $(-4.5,\,{}$ {glue:text}`t_path_y` $)$, outside
the occulter's dashed outline, so its core is bright. The unit of position is
$\lambda_0/D$: one wavelength $\lambda_0$ divided by the pupil diameter, which is
about the width of the core. The values are chosen for speed and for effect. The
whole detector is only {glue:text}`t_npix` pixels across, so many images are cheap,
and an occulter radius of {glue:text}`t_occ` $\lambda_0/D$ is large enough that
the planet crosses a region where the image changes shape, not only position.

## What does one detector pixel see as the planet crosses?

Pick one detector pixel to watch, the yellow square in panel 4. It is the pixel
whose centre, ({glue:text}`t_probe_x`, {glue:text}`t_probe_y`) $\lambda_0/D$, is
nearest $(3.1,\,0.6)$, outside the occulter, where the planet's core is bright
when it arrives. The planet moves along the pink line, from left to right at
$s_y={}$ {glue:text}`t_path_y`, and the pixel's brightness is recorded at every
position.

```{admonition} Predict before reading on
:class: tip
Along the pink line, near which planet position $s_x$ is the yellow pixel
brightest: near $-3$, near 0 or near 3?
```

```{code-cell} python
:tags: [hide-input]
s_path = np.concatenate([np.full(3, -6.0), np.linspace(-6.0, 6.0, 37), np.full(6, 6.0)])
path = np.stack([s_path, np.full_like(s_path, PATH_Y)], axis=1)
images = np.asarray(toy.psf(path))
trace = images[:, row, col]
s_fine = np.linspace(-6, 6, 961)
fine = np.asarray(toy.psf(np.stack([s_fine, np.full_like(s_fine, PATH_Y)], 1)))[:, row, col]
LOW = 1e-11

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1, 1.25]})
image = viz.plot_read(images[0], pixscale_lod=toy.pixscale_lod, ax=left, vmin=1e-9, vmax=0.5)
left.add_patch(plt.Circle((0, 0), toy.occulter_lod, fill=False, color=SCENERY, lw=1, ls="--"))
left.plot(x_pix[col], x_pix[row], "s", mfc="none", mec=pal.yellow, ms=7, mew=1.5)
(planet,) = left.plot([], [], "+", color=pal.pink, ms=10, mew=1.5)
left.set_xlim(-8, 8)
left.set_ylim(-8, 8)
right.set_xlim(-6, 6)
right.set_ylim(LOW, 1)
right.set_yscale("log")
right.set_xlabel(r"planet position $s_x$ ($\lambda_0/D$)")
right.set_ylabel("probe pixel intensity")
(curve,) = right.plot([], [], color=pal.yellow, lw=2)
(dot,) = right.plot([], [], "o", color=pal.yellow, ms=6)
right.axvspan(-toy.occulter_lod, toy.occulter_lod, color=ep.style.neutral(0.15), lw=0)
right.text(0, 3e-2, "planet behind the occulter", ha="center", color=ep.style.neutral(0.55),
           fontsize=9)


def draw(fig, k):
    image.update(images[k])
    planet.set_data([path[k, 0]], [path[k, 1]])
    shown = s_fine <= s_path[k]
    curve.set_data(s_fine[shown], fine[shown])
    dot.set_data([s_path[k]], [trace[k]])


anim = ep.render.animate(fig, draw, range(len(s_path)), fps=10)
clip(anim)
```

In the clip the planet crosses the field from left to right. On the left the
detector image changes character: a bright core outside the occulter, a faint
leak while the planet is behind the occulter, then the core again. On the right
the probe pixel's intensity is drawn as the planet moves, on a logarithmic axis
so that the faint parts show.

```{code-cell} python
:tags: [hide-input]
fig = plt.figure(figsize=(9, 6.4), constrained_layout=True)
grid = fig.add_gridspec(2, 6, height_ratios=[1, 0.9])
for j, k in enumerate((8, 22, 36)):
    ax = fig.add_subplot(grid[0, 2 * j:2 * j + 2])
    viz.plot_read(images[k], pixscale_lod=toy.pixscale_lod, ax=ax, vmin=1e-9, vmax=0.5,
                  colorbar=False)
    ax.add_patch(plt.Circle((0, 0), toy.occulter_lod, fill=False, color=SCENERY, lw=1, ls="--"))
    ax.plot(x_pix[col], x_pix[row], "s", mfc="none", mec=pal.yellow, ms=6, mew=1.2)
    ax.plot(path[k, 0], path[k, 1], "+", color=pal.pink, ms=9, mew=1.5)
    ax.set_title(rf"$s_x = {path[k, 0]:+.1f}$")
    ax.set_xlabel("")
    if j:
        ax.set_ylabel("")
lin = fig.add_subplot(grid[1, :3])
lin.plot(s_fine, fine, color=pal.yellow, lw=2)
lin.axvspan(-toy.occulter_lod, toy.occulter_lod, color=ep.style.neutral(0.15), lw=0)
lin.set_xlabel(r"planet position $s_x$ ($\lambda_0/D$)")
lin.set_ylabel("probe pixel, linear axis")
lin.set_title("what the pixel does", fontsize=10, color=SCENERY)
log = fig.add_subplot(grid[1, 3:])
log.semilogy(s_fine, fine, color=pal.yellow, lw=2)
log.axvspan(-toy.occulter_lod, toy.occulter_lod, color=ep.style.neutral(0.15), lw=0)
log.set_ylim(LOW, 1)
log.set_xlabel(r"planet position $s_x$ ($\lambda_0/D$)")
log.set_ylabel("same curve, logarithmic axis")
log.set_title("what the faint structure does", fontsize=10, color=SCENERY)
plt.show()

peak_at = s_fine[np.argmax(fine)]
dips = np.where((fine[1:-1] < fine[:-2]) & (fine[1:-1] < fine[2:]))[0] + 1
deep = s_fine[dips][np.abs(s_fine[dips]) < 6]
num("t_peak_at", f"{peak_at:.1f}")
num("t_peak", f"{fine.max():.2f}")
num("t_ndips", f"{len(deep)}")
num("t_dip_gap", f"{np.mean(np.diff(deep)):.1f}")
num("t_decades", f"{np.log10(fine.max() / fine.min()):.0f}")
```

The three moments above show the planet before, behind and past the occulter, and
the yellow square is the pixel. Look at the left curve below them: the pixel is
dark until the planet's core arrives, peaks at {glue:text}`t_peak` when the planet is
at $s_x=${glue:text}`t_peak_at`, and goes dark again, so the prediction is the
planet position nearest the pixel. The right curve is the same data on a
logarithmic axis. It shows a faint structure that the linear axis hides: the
curve spans {glue:text}`t_decades` orders of magnitude, and it dips
{glue:text}`t_ndips` times in the range plotted, about {glue:text}`t_dip_gap`
$\lambda_0/D$ apart. Because the curve is an intensity, a field squared, a dip
on a logarithmic axis is a place where the field passes through zero.

## What this page established

:::{admonition} What perfect interpolation needs
:class: important

1. **A band limit.** The signal is a pixel's brightness against planet position:
   a bright bump with faint, regular dips. Its bandwidth is the open question.
2. **Dense enough samples.** $\rho=2f_{\max}<1$.
3. **Weights a program can afford.** The Kaiser-sinc kernel.
:::

The toy has a probe pixel, a path for the planet, and a curve of the pixel's
brightness against the planet's position. The next page uses this pixel to count
what one read of a two-dimensional grid costs, and {doc}`band-limits` then asks how
fast the curve can change.

Next: {doc}`two-dimensional-read`.
