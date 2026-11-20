---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  render_markdown_format: myst
---

# How can one quadrant serve the whole field?

A grid that covers planets in all four quadrants costs four times the
propagations of a grid that covers one. Many coronagraphs are mirror
symmetric, so in principle a designer could deliver one quadrant and let the
reader supply the rest. This page asks what the reader does with such a grid,
what must be true of the optics for the shortcut to be honest, and how many
nodes the quadrant needs. It uses the toy coronagraph of
{doc}`toy-coronagraph` and the two-dimensional read of
{doc}`two-dimensional-read`.

:::{admonition} What perfect interpolation needs
:class: important

1. **A band limit.** The signal has no cosine above a known bandwidth
   $f_{\max}$. *({doc}`band-limits`)*
2. **Dense enough samples.** $\rho=2f_{\max}<1$, set by the node spacing $h$.
   **This page counts the nodes that spacing needs, and stores a quarter of
   them.** *({doc}`../foundations/sampling`)*
3. **Weights a program can afford.** The Kaiser-sinc kernel. *({doc}`kernel`)*
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
from IPython.display import Video
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from myst_nb import glue

import eyepiece as ep
import yippy.sinc as sc
from yippy.sinc.geometry import source_grid
from yippy.sinc.reader import read_batch
from yippy.sinc import viz
from yippy.sinc.kernel import KernelProfile
import toy_coronagraph as tc

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette
SCENERY = ep.style.neutral(0.45)
H = 0.38

_clips = iter(range(100))


def clip(anim):
    """Embed an animation as an mp4 with the browser's own player controls."""
    path = Path(tempfile.gettempdir()) / f"yippy-sinc-docs-quadrant-{next(_clips)}.mp4"
    anim.video(path, dpi=200)
    return Video(str(path), embed=True, html_attributes="controls loop muted playsinline")


def num(name, text):
    """Glue a plain string."""
    glue(name, text, display=False)


def tex(name, value):
    """Glue a number as math with a power of ten."""
    mantissa, exponent = f"{value:.1e}".split("e")
    glue(name, f"${mantissa}\\times10^{{{int(exponent)}}}$", display=False)


toy = tc.ToyCoronagraph()
delivery = tc.quadrant_delivery(toy)
admission = sc.admit_grid(**delivery)
reader = tc.reader_for(delivery)
lattice = delivery["declared_lattice"]
pix = np.asarray(toy.detector_x)
pitch = toy.pixscale_lod
extent = (pix[0] - pitch / 2, pix[-1] + pitch / 2) * 2


def negative_columns(s):
    """Count the 32 stencil columns that lie at a negative source position."""
    floor = np.floor(s / H)
    return int(np.sum(floor + np.arange(-15, 17) < 0))


num("q_nodes", f"{lattice.shape[1]} by {lattice.shape[0]}")
num("q_h", f"{H}")
num("q_neg_plus", f"{negative_columns(0.2)}")
num("q_neg_minus", f"{negative_columns(-0.2)}")
num("q_t", f"{0.2 / H:.2f}")
```

## What does a read near the edge of a quadrant need?

The toy has a circular pupil, a circular occulter and a circular Lyot stop, so
it is symmetric about both axes. Its quadrant grid stores
{glue:text}`q_nodes` nodes at spacing {glue:text}`q_h` $\lambda_0/D$, all with
$s_x\ge0$ and $s_y\ge0$, starting exactly at the origin.

```{admonition} Predict before reading on
:class: tip
A planet sits at $s_x=+0.2\,\lambda_0/D$, inside the stored quadrant. Does
its read need any data from negative source positions? If so, roughly what
fraction of its stencil?
```

```{dropdown} Answer
Yes, about half. Its position is {glue:text}`q_t` samples from the origin
(0.2 divided by the spacing {glue:text}`q_h`), so its stencil runs from node
$-15$ to node $16$ along $x$:
{glue:text}`q_neg_plus` of its 32 columns of nodes lie at negative $s_x$,
where nothing was stored.
```

The figure shows that stencil for a planet at $(0.2,\,3.0)$. Marker area is
the size of each tap's weight and the color its sign, as on
{doc}`two-dimensional-read`. On the left the stencil is drawn where the
kernel puts it, and the hollow markers at negative positions are nodes that
were never stored. On the right each of those taps has been **folded**: moved
onto the stored node at the mirrored position $(|i|,|j|)$ that it reads
instead.

```{code-cell} python
:tags: [hide-input]
query = np.array([0.2, 3.0])
fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True, sharey=True)
for ax, fold, title in ((axes[0], 0.0, "as the kernel places it"),
                        (axes[1], 1.0, "folded onto stored nodes")):
    viz.plot_stencil_2d(query, lattice, profile=KernelProfile(), quadrant=True,
                        fold=fold, max_area=700.0, ax=ax)
    ax.axvline(0, color=SCENERY, lw=1, ls="--")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-3.5, 9.5)
    ax.set_title(title + f": {negative_columns(query[0])} of 32 columns at $s_x<0$"
                 if fold == 0 else title, fontsize=10)
axes[1].set_ylabel("")
axes[0].text(-6.3, -3.2, "mirrored nodes, no data", color=SCENERY, fontsize=8)
plt.show()
```

Folding is possible only if reversing the source's $x$ position reverses the
detector columns, and reversing $y$ reverses the rows. The next section tests
exactly that.

## Is the flipped copy the same image as the real one?

A fold reads a stored image flipped. It is exact only if the image the
optics would make at the negative node is the flipped image of the stored one.
For the toy that can be checked directly, because the toy can propagate any
position. The top row shows the images for a planet at $(\pm3,\pm2)$. The
middle row flips the image at $(3,2)$ by the matching axis. The bottom row is
the difference between the two, as a fraction of the peak, on a scale that
runs down to float64 rounding.

```{code-cell} python
:tags: [hide-input]
spots = [(3.0, 2.0), (-3.0, 2.0), (3.0, -2.0), (-3.0, -2.0)]
flips = [lambda a: a, lambda a: a[:, ::-1], lambda a: a[::-1, :], lambda a: a[::-1, ::-1]]
true = [np.asarray(toy.psf(np.array([s])))[0] for s in spots]
stored = true[0]
fig, axes = plt.subplots(3, 4, figsize=(10, 7.6), constrained_layout=True)
worst = 0.0
for j, (s, img, flip) in enumerate(zip(spots, true, flips)):
    copy = flip(stored)
    residual = np.abs(img - copy) / img.max()
    worst = max(worst, residual.max())
    axes[0, j].imshow(img, origin="lower", extent=extent, norm=LogNorm(1e-9, 0.5),
                      cmap=hwostyle.cmaps.intensity)
    axes[0, j].set_title(rf"planet at $({s[0]:+.0f},{s[1]:+.0f})$", fontsize=10)
    axes[1, j].imshow(copy, origin="lower", extent=extent, norm=LogNorm(1e-9, 0.5),
                      cmap=hwostyle.cmaps.intensity)
    axes[1, j].set_title(["the stored image", "flipped in $x$", "flipped in $y$",
                          "flipped in both"][j], fontsize=10)
    last = axes[2, j].imshow(np.maximum(residual, 1e-18), origin="lower", extent=extent,
                             norm=LogNorm(1e-18, 1e-6), cmap="magma")
    for row in range(3):
        axes[row, j].set_xticks([])
        axes[row, j].set_yticks([])
axes[0, 0].set_ylabel("true image")
axes[1, 0].set_ylabel("flipped stored image")
axes[2, 0].set_ylabel("difference / peak")
fig.colorbar(last, ax=axes[2, :], shrink=0.8, label="difference / peak")
plt.show()
tex("q_mirror_worst", worst)
```

The three flipped copies agree with the three true images to
{glue:text}`q_mirror_worst` of the peak, which is float64 rounding. For this
optical train, then, a mirrored node is not an approximation of the real one.

## Does a read across the axis show a seam?

The left panel of the clip is the stencil view above, now with the query
moving. It first holds a read at $s_x=1.5$ and folds every column at a
negative $x$ onto the stored column it reads. Then the query moves through
the axis to $s_x=-1.5$ at fixed $s_y=3$, with the folded stencil reading
stored nodes throughout. The right panel is the image the reader returns.

```{code-cell} python
:tags: [hide-input]
# A sweep: the stencil changes which stored nodes it reads as the query crosses
# the axis, and the image must show no seam there, which separate stills cannot show.
sx = np.concatenate([np.full(10, 1.5), np.linspace(1.5, -1.5, 31), np.full(5, -1.5)])
fold = np.concatenate([np.zeros(3), np.linspace(0, 1, 8)[1:], np.ones(36)])
path = np.stack([sx, np.full_like(sx, 3.0)], axis=1)
reads = read_batch(reader, jnp.asarray(path), jnp.zeros(len(sx), dtype=jnp.int32))
images = np.asarray(reads.intensity)

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
stencil_view = viz.plot_stencil_2d(path[0], lattice, profile=KernelProfile(), quadrant=True,
                                   max_area=250.0, ax=left)
ep.motion.carried_tag(left, "the stencil of a read")
left.axvline(0, color=SCENERY, lw=1, ls="--")
left.set_xlim(-9, 9)
left.set_ylim(-5, 12.5)
left.text(11.2, 11.8, "stored nodes", ha="right", color=ep.style.neutral(0.7), fontsize=8)
left.text(-8.7, -4.3, "mirrored nodes", color=SCENERY, fontsize=8)
psf = viz.plot_read(images[0], pixscale_lod=toy.pixscale_lod, ax=right, vmin=1e-9, vmax=0.5)
right.axvline(0, color=SCENERY, lw=1, ls="--")
right.set_xlim(-6, 6)
right.set_ylim(-3, 9)


def draw(fig, k):
    stencil_view.update(path[k], fold=fold[k])
    psf.update(images[k])
    left.set_title(rf"$s_x = {sx[k]:+.2f}$: {negative_columns(sx[k])} of 32 columns at $s_x<0$",
                   fontsize=10)


anim = ep.render.animate(fig, draw, range(len(sx)), fps=10)
clip(anim)
```

The planet's core moves smoothly through the axis. The still frames below
show before, on and after the axis, the folded stencil above and the read
below. The middle read, at $s_x=0$ from a stencil that is almost half
mirrored nodes, is as clean as the other two.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
for j, k in enumerate((10, 25, 41)):
    viz.plot_stencil_2d(path[k], lattice, profile=KernelProfile(), quadrant=True,
                        fold=1.0, max_area=250.0,
                        ax=axes[0, j])
    axes[0, j].axvline(0, color=SCENERY, lw=1, ls="--")
    axes[0, j].set_xlim(-9, 9)
    axes[0, j].set_ylim(-5, 12.5)
    axes[0, j].set_title(rf"$s_x = {sx[k]:+.2f}$: {negative_columns(sx[k])} of 32 "
                         r"columns at $s_x<0$", fontsize=9)
    viz.plot_read(images[k], pixscale_lod=toy.pixscale_lod, ax=axes[1, j], vmin=1e-9,
                  vmax=0.5, colorbar=False)
    axes[1, j].axvline(0, color=SCENERY, lw=1, ls="--")
    axes[1, j].set_xlim(-6, 6)
    axes[1, j].set_ylim(-3, 9)
plt.show()
```

A smooth picture could still be the wrong picture, so the next figure
compares the reader with a direct propagation of the toy. The planet sweeps
from $s_x=-3$ to $+3$ at $s_y=3$. On the left, one detector pixel, the one
nearest $(1.5,\,3.0)$, drawn from the truth as a line and from the reader as
dots. On the right, the largest difference over the whole image at each
position, as a fraction of that image's peak.

```{code-cell} python
:tags: [hide-input]
sweep_x = np.linspace(-3, 3, 121)
sweep = np.stack([sweep_x, np.full_like(sweep_x, 3.0)], axis=1)
truth_imgs = np.asarray(toy.psf(sweep))
read_imgs = np.asarray(
    read_batch(reader, jnp.asarray(sweep), jnp.zeros(len(sweep), dtype=jnp.int32)).intensity
)
err = np.abs(read_imgs - truth_imgs).max(axis=(1, 2)) / truth_imgs.max(axis=(1, 2))
pc = int(np.argmin(np.abs(pix - 1.5)))
pr = int(np.argmin(np.abs(pix - 3.0)))
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
a.semilogy(sweep_x, truth_imgs[:, pr, pc], color=pal.yellow, lw=2, label="direct propagation")
a.semilogy(sweep_x[::4], read_imgs[::4, pr, pc], "o", color=pal.cyan, ms=4,
           label="quadrant read")
a.axvline(0, color=SCENERY, lw=1, ls="--")
a.set_xlabel(r"planet position $s_x$ ($\lambda_0/D$)")
a.set_ylabel("one pixel's intensity")
a.legend(fontsize=8, frameon=False)
b.semilogy(sweep_x, np.maximum(err, 1e-18), color=pal.pink, lw=2)
b.set_ylim(1e-9, 1e-5)
b.axvline(0, color=SCENERY, lw=1, ls="--")
b.set_xlabel(r"planet position $s_x$ ($\lambda_0/D$)")
b.set_ylabel("largest difference / image peak")
plt.show()
tex("q_read_err", err.max())
tex("q_read_err_axis", err[np.argmin(np.abs(sweep_x))])
```

The quadrant read follows the truth across the axis, with a largest
difference of {glue:text}`q_read_err` of the image peak over the sweep. No
seam shows at $s_x=0$, where the difference is
{glue:text}`q_read_err_axis`, and the error is as flat across the axis as
elsewhere. It is the kernel's interpolation error, and the fold adds none of
its own.

Mechanically, a tap at signed node index $(i,j)$ reads the stored image at
$(|i|,|j|)$, reversing the detector columns when $i<0$ and the rows when
$j<0$. The kernel weights are the same as for a full grid, so a quadrant read
is algebraically identical to a full-grid read of the symmetric extension.

The reversal is an exact mirror only if the detector is centered correctly.
The figure shows the rule: the center sits at pixel index $(n-1)/2$, between
the two middle pixels when $n$ is even and on the middle pixel when $n$ is
odd. Pixel $i$ then swaps with pixel $n-1-i$ and the mirror line passes
through the center in both cases. A detector centered on a pixel corner, or
one pixel off, makes the flipped image sit one pixel away from the real one.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(2, 1, figsize=(8, 2.8), constrained_layout=True)
for ax, n in zip(axes, (6, 5)):
    for i in range(n):
        ax.add_patch(Rectangle((i, 0), 1, 1, fill=False, ec=SCENERY, lw=1.2))
        ax.text(i + 0.5, 0.5, str(i), ha="center", va="center", fontsize=10)
    centre = (n - 1) / 2 + 0.5
    ax.axvline(centre, color=pal.pink, lw=2)
    ax.annotate("", xy=(n - 0.5, 1.3), xytext=(0.5, 1.3),
                arrowprops=dict(arrowstyle="<->", color=pal.cyan, lw=1.5))
    ax.text(centre + 0.08, -0.3, rf"center $=(n-1)/2={(n - 1) / 2:g}$", color=pal.pink,
            fontsize=9, va="top")
    ax.text(centre, 1.5, rf"$n={n}$: pixel $i$ swaps with pixel {n - 1}$-i$", ha="center",
            color=pal.cyan, fontsize=9)
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-0.9, 2.0)
    ax.axis("off")
plt.show()
```

## When is a mirrored node a faithful copy of the real one?

The shortcut is only as good as the symmetry. Two geometric mirrors of the
optics suffice: if the pupil, the occulter and the Lyot stop are all
symmetric about the $x$ axis and about the $y$ axis, each source mirror maps
to the matching detector mirror.

One geometric mirror can also suffice, through a less obvious route. Suppose
the entrance amplitude, the Lyot stop and the focal-plane mask are real, and
the mask is **centrosymmetric**: unchanged by inversion through its center, as
a circular disk is. Then inverting the source and the detector together
leaves the intensity unchanged. The dropdown gives the argument in steps.

```{dropdown} Why inversion leaves the intensity unchanged
Write $E(\mathbf{x};\mathbf{s})$ for the field at detector position
$\mathbf{x}$ from a source at $\mathbf{s}$. The claim is
$E(-\mathbf{x};-\mathbf{s})=E^*(\mathbf{x};\mathbf{s})$.

1. Reversing the source's sign reverses the sign of the tilt. The phase ramp
   across the pupil becomes its complex conjugate.
2. The pupil amplitude is real, so the whole pupil field at $-\mathbf{s}$ is
   the conjugate of the pupil field at $\mathbf{s}$.
3. A Fourier transform of a conjugated function is the conjugate of the
   transform evaluated at the inverted coordinate. The field in the focal
   plane at $-\mathbf{s}$ is therefore the conjugate of the field at
   $\mathbf{s}$, inverted through the center.
4. The mask is real and centrosymmetric, so multiplying by it commutes with
   both the conjugation and the inversion.
5. The same step through the second transform, with a real Lyot stop, carries
   the conjugation and the inversion to the detector.

The detector field at $-\mathbf{s}$ is the conjugate of the field at
$\mathbf{s}$, inverted through the center. The intensity is the squared
magnitude, which a conjugate does not change.
```

Inversion is a flip in $x$ followed by a flip in $y$. If one flip is also a
geometric mirror, the other follows. The figure tests this on the toy for a
planet at $(1.14,\,0.76)$. Each row decenters the pupil, each column after
the first shows the difference between a true image and a flipped copy of the
image at the opposite source position, on the scale of the earlier figure.
Shifting the pupil along $y$ alone leaves $x$ as a mirror and the intensity
passes all three tests, including the $y$ flip that the geometry no longer
supports. A diagonal shift removes both mirrors and the two flips fail, while
the inversion still passes.

```{code-cell} python
:tags: [hide-input]
src = np.array([1.14, 0.76])
tests = [("$x$ flip", lambda a: a[:, ::-1], (-1, 1)),
         ("$y$ flip", lambda a: a[::-1, :], (1, -1)),
         ("inversion", lambda a: a[::-1, ::-1], (-1, -1))]
fig, axes = plt.subplots(2, 4, figsize=(10, 5.4), constrained_layout=True)
results = {}
for r, shift in enumerate(((0.0, 0.1), (0.1, 0.1))):
    t = tc.ToyCoronagraph(pupil_shift=shift)
    uu, vv = (np.asarray(c) for c in t.pupil_coords)
    axes[r, 0].imshow(np.asarray(t.pupil), origin="lower", cmap="gray", vmin=0, vmax=1.4,
                      extent=(uu.min(), uu.max(), vv.min(), vv.max()))
    axes[r, 0].axvline(0, color=pal.pink, lw=1, ls="--")
    axes[r, 0].axhline(0, color=pal.pink, lw=1, ls="--")
    axes[r, 0].set_title(f"pupil shifted by {shift}", fontsize=9)
    base = np.asarray(t.psf(src[None]))[0]
    for c, (name, flip, sign) in enumerate(tests, start=1):
        other = np.asarray(t.psf((src * np.array(sign))[None]))[0]
        resid = np.abs(other - flip(base)) / base.max()
        results[(r, name)] = resid.max()
        shown = axes[r, c].imshow(np.maximum(resid, 1e-18), origin="lower", extent=extent,
                                  norm=LogNorm(1e-18, 1), cmap="magma")
        axes[r, c].set_title(f"{name}: largest {resid.max():.0e}", fontsize=9)
    for c in range(4):
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
fig.colorbar(shown, ax=axes[:, 1:], shrink=0.8, label="difference / peak")
plt.show()
tex("q_shift_y_xflip", results[(0, "$x$ flip")])
tex("q_shift_y_yflip", results[(0, "$y$ flip")])
tex("q_shift_d_xflip", results[(1, "$x$ flip")])
tex("q_shift_d_inv", results[(1, "inversion")])
```

With the shift along $y$ only, the $x$ flip and the $y$ flip differ from the
truth by {glue:text}`q_shift_y_xflip` and {glue:text}`q_shift_y_yflip`. With
the diagonal shift the $x$ flip is off by {glue:text}`q_shift_d_xflip`, a
failure that is easy to see, and the inversion still holds at
{glue:text}`q_shift_d_inv`. A quadrant grid needs the two flips, so the
diagonal shift is refused, and {doc}`../practice/designers` runs this same
pair of decenters through admission.

The public LUVOIR-A apodized pupil Lyot coronagraph (APLC) pupil used on
{doc}`../validation` has a mirror about one axis only, and the conjugation
argument is what makes the APLC two-axis symmetric in intensity.

Many things break the argument: aberrations or deformable mirror phases,
misalignment, polarization aberrations, and complex focal-plane masks such as
those of hybrid Lyot and vortex coronagraphs. A designer therefore declares
which argument applies to the actual optical state, and yippy.sinc checks it.

## How does the reader know the symmetry is real?

The declaration alone is never trusted. A quadrant delivery carries
independent propagations at negative positions (an $x$-negative, a
$y$-negative and a both-negative counterpart for each product), and
admission compares each with the flipped stored image of its mirrored node.
It also checks every stored image on an axis against its own flip, since a
source on the $x$ axis must produce an image symmetric under row reversal.
The tolerance is $10^{-6}$ of the local peak. A looser tolerance cannot be
requested, and caller-supplied pass flags carry no weight.

```{code-cell} python
:tags: [hide-input]
receipt = admission.receipt
axis = np.asarray(receipt.axis_relative, dtype=float)
n_axis = len(receipt.axis_exact)
n_exact = int(sum(receipt.axis_exact))
print("admitted:", admission.admitted)
print("mirror residuals / local peak:", [f"{v:.1e}" for v in receipt.mirror_local_peak])
print(f"axis images: {n_exact} of {n_axis} identical to their flip bit for bit, "
      f"largest relative residual of the rest {np.nanmax(axis):.1e}")
num("q_n_axis", f"{n_axis}")
num("q_n_exact", f"{n_exact}")
num("q_n_rest", f"{n_axis - n_exact}")
tex("q_axis_worst", float(np.nanmax(axis)))
tex("q_mirror_max", float(max(receipt.mirror_local_peak)))
```

Admitted means every check passed. The three mirror residuals are at most
{glue:text}`q_mirror_max`. Among the {glue:text}`q_n_axis` images on the two
axes, {glue:text}`q_n_exact` equals its flip exactly, an accident of
rounding. The other {glue:text}`q_n_rest` differ from their flips by at most
{glue:text}`q_axis_worst`, which is rounding too, and each passes on the
relative residual. For the toy every residual sits at the level of float64
rounding.

A finite set of checks cannot prove the symmetry everywhere, but it catches
the common mistakes: a decentered pupil, a mislabeled axis or a flipped
detector convention.

## How many nodes does the quadrant need?

The spacing $h$ comes from the bandwidth, as {doc}`band-limits` showed. The
node count then follows from the region to serve. A read reaches 16 nodes
past the start of its own cell, so a quadrant grid that serves reads up to,
but not including, $R$ along an axis needs $\lceil R/h\rceil+16$ nodes, with
indices $0$ through $\lceil R/h\rceil+15$. The first figure counts them along
one axis for the physicaloptix starter delivery on {doc}`../validation`,
whose spacing is $h=0.38$ and whose served edge along $x$ is 4.18.

```{code-cell} python
:tags: [hide-input]
from optixstuff.grids import Frame
starter = source_grid(jnp.zeros(2), jnp.full(2, H), (31, 27), Frame(500.0, 6.0, "circumscribed"))
ny, nx = starter.shape
served_x = (nx - 16) * H
served_y = (ny - 16) * H
fig, ax = plt.subplots(figsize=(10, 2.2), constrained_layout=True)
idx = np.arange(nx)
ax.scatter(idx[: nx - 16] * H, np.zeros(nx - 16), color=pal.cyan, s=30, zorder=3,
           label=f"{nx - 16} nodes up to the served edge")
ax.scatter(idx[nx - 16:] * H, np.zeros(16), color=SCENERY, s=30, zorder=3,
           label="16 nodes of stencil margin")
ax.axvline(served_x, color=pal.pink, lw=1.5, ls="--", zorder=1)
ax.text(served_x - 0.1, 0.55, f"served edge {served_x:.2f}", color=pal.pink,
        ha="right", fontsize=9)
ax.set_xlim(-0.3, (nx - 1) * H + 0.3)
ax.set_ylim(-0.6, 1.0)
ax.set_yticks([])
ax.set_xlabel(r"node position along $x$ ($\lambda_0/D$)")
ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)
plt.show()
num("q_nx", f"{nx}")
num("q_ny", f"{ny}")
num("q_total", f"{nx * ny}")
num("q_rx", f"{served_x:.2f}")
num("q_ry", f"{served_y:.2f}")
num("q_kx", f"{nx - 16}")
num("q_ky", f"{ny - 16}")
```

Along $x$ the {glue:text}`q_kx` cyan nodes reach the served edge
{glue:text}`q_rx`, and the 16 gray nodes are the margin that its stencils
need. The same count along $y$ gives {glue:text}`q_ky` plus 16, or
{glue:text}`q_ny` nodes, and a served edge of {glue:text}`q_ry`. The
rectangle is therefore {glue:text}`q_nx` wide and {glue:text}`q_ny` tall,
{glue:text}`q_total` nodes in all. The label counts nodes along $x$ and
then along $y$, while the code's lattice shape is ordered $(n_y,n_x)$.

The second figure shows the grid itself. The cyan dots are the stored
quadrant. The gray box is the region whose mirrored stencils are all
complete, so reads are supported anywhere in it.

```{code-cell} python
:tags: [hide-input]
gx, gy = np.meshgrid(np.arange(nx) * H, np.arange(ny) * H)
fig, ax = plt.subplots(figsize=(6, 6.4), constrained_layout=True)
ax.add_patch(Rectangle((-served_x, -served_y), 2 * served_x, 2 * served_y,
                       color=ep.style.neutral(0.15), zorder=0))
ax.scatter(gx.ravel(), gy.ravel(), s=8, color=pal.cyan, zorder=3)
ax.axhline(0, color=SCENERY, lw=1, ls="--")
ax.axvline(0, color=SCENERY, lw=1, ls="--")
ax.text(-served_x + 0.15, -served_y + 0.25, "served region (mirrored)", color=SCENERY,
        fontsize=9)
ax.text(0.2, (ny - 1) * H + 0.2, f"stored nodes: {nx} by {ny}", color=pal.cyan, fontsize=9)
ax.set_aspect("equal")
ax.set_xlim(-served_x - 0.5, (nx - 1) * H + 0.5)
ax.set_ylim(-served_y - 0.5, (ny - 1) * H + 1.0)
ax.set_xlabel(r"$s_x$ ($\lambda_0/D$)")
ax.set_ylabel(r"$s_y$ ($\lambda_0/D$)")
plt.show()
```

The starter rehearsal declares only the corner cell
$3.8\le s_x<4.18$, $5.32\le s_y<5.70$ as its served region, which is also
where all its checks lie, so reads elsewhere in the box refuse for this
delivery and would be unqualified if a delivery declared them.

The spacing 0.38 and the count follow a planning pupil width $L=1$ with
$\rho=0.76$. The starter's declaration is smaller, with widths
0.946 by 0.982 $D$, so its margins are $\rho=2h\sigma_{\max}L$, computed
below.

```{code-cell} python
:tags: [hide-input]
rho_x = 2 * H * 0.946
rho_y = 2 * H * 0.982
print(f"rho along x: {rho_x:.3f}, along y: {rho_y:.3f}")
num("q_rho_x", f"{rho_x:.3f}")
num("q_rho_y", f"{rho_y:.3f}")
num("q_rho_plan", f"{2 * H:.2f}")
```

The margins are {glue:text}`q_rho_x` along $x$ and {glue:text}`q_rho_y` along
$y$, below the {glue:text}`q_rho_plan` that $L=1$ gives, so the grid is
slightly finer than it needs to be. {doc}`../practice/designers` turns this
chain into the choices a designer makes.

## Try it

1. A planet sits at $s_x=-0.2$, $s_y=1.0$ on the toy grid. Before computing,
   predict how many of its 32 stencil columns lie at negative $s_x$, then
   check with `negative_columns` from the setup cell.
   ```{dropdown} Hint
   $t=-0.2/0.38=-0.53$, so the floor is $-1$ and the columns run from $-16$
   to $15$: 16 negative columns, one more than at $s_x=+0.2$.
   ```
2. Which of these would break the conjugation argument for the toy: a Lyot
   stop shifted along $y$, a small focus aberration added to the pupil, a
   pupil scaled to 0.95 $D$?
   ```{dropdown} Hint
   The focus aberration, because it makes the pupil field complex. The
   shifted stop stays real and keeps a mirror axis, and a scaled pupil keeps
   both mirrors (it changes the bandwidth instead, see
   {doc}`band-limits`).
   ```
3. A delivery serves reads up to $s_x<3.0$ and $s_y<3.0$ at $h=0.38$. How many
   nodes does the quadrant need along each axis?
   ```{dropdown} Hint
   $\lceil3.0/0.38\rceil=8$, so $8+16=24$ nodes along each axis.
   ```

## What this page established

:::{admonition} What perfect interpolation needs
:class: important

1. **A band limit.** The signal has no cosine above a known bandwidth
   $f_{\max}$. *({doc}`band-limits`)*
2. **Dense enough samples.** $\rho=2f_{\max}<1$. **The spacing sets the node
   count, $\lceil R/h\rceil+16$ per axis, and one quadrant stored with a
   verified symmetry serves all four.** *({doc}`../foundations/sampling`)*
3. **Weights a program can afford.** The Kaiser-sinc kernel. *({doc}`kernel`)*
:::

A quadrant grid reads negative positions by flipping stored images. The flip
is exact given the declared two-axis symmetry of the optics, and admission
verifies that symmetry against independent propagations to $10^{-6}$ of the
local peak before any read. With bandwidth, spacing, node count and symmetry
settled, the remaining question is what the numerical contract states about
the taps.

Next: {doc}`../reference/numerical-contract`.
