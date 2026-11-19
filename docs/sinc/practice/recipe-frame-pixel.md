---
file_format: mystnb
kernelspec:
  name: python3
---

# Where does the star sit, and what does one pixel hold?

The planner of {doc}`delivery-recipe` gave the positions to propagate. This
page fixes how each propagated image is framed and what its pixel values mean,
which are the two choices that the checks inside the delivery cannot catch when
they are consistently wrong: every image can agree with every other and still
put the star in the wrong place or carry the wrong units.

:::{admonition} The question of this page
:class: important

How do you place the star and define a pixel value so that a consistent
mistake cannot hide? Step 2 places the star and tests it against the sky, and
step 3 defines what a pixel holds. The close restates both as things to do.
:::

```{code-cell} python
:tags: [remove-cell]
import sys

sys.path.insert(0, "../scripts")
import equinox as eqx
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown
from matplotlib.patches import Polygon
from myst_nb import glue
from validation_tables import load

import yippy.sinc as sc
import toy_coronagraph as tc

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette
toy = tc.ToyCoronagraph()
center = (toy.n_pix - 1) / 2


def tex(value, digits=2):
    """Format a number as inline math a x 10^b."""
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    return Markdown(rf"${mantissa}\times10^{{{int(exponent)}}}$")
```

(recipe-frame)=
## Step 2: Where does the star sit, and how do you know?

Every image has the same shape, pixel pitch and center, and the star is at
the center of the frame, which is the point $((n_x-1)/2,\,(n_y-1)/2)$ in
zero-based pixel coordinates (the manifest key `detector_center_xy`). For an
even dimension that is between the two middle pixels, and for an odd dimension
it is on the middle pixel. The starter's 128 pixels put the center at 63.5,
and the toy's 96 pixels ({ref}`toy-in-numbers`) put it at 47.5. Increasing
column follows positive source $x$ and increasing row follows positive source
$y$. The shape must be square, and `PIXSCALE` is in $\lambda_0/D$ per pixel
([header keywords][schema-fits]).

The figure shows the toy image of a source at $(3, 0)$ with the rows drawn
upward. Look at the white cross, which is the frame center where the star
sits, and the cyan ring, which is where the core of the planet must land:
$3/0.25=12$ pixels to the right of the star, because the source is 3
$\lambda_0/D$ along $+x$ and each pixel is 0.25 $\lambda_0/D$.

```{code-cell} python
:tags: [hide-input]
source = (3.0, 0.0)
col_expected = center + source[0] / toy.pixscale_lod
row_expected = center + source[1] / toy.pixscale_lod
image = np.asarray(toy.psf(jnp.array(source)))
fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
ax.imshow(np.log10(image + 1e-12), origin="lower", cmap="magma", vmin=-9, vmax=0)
ax.scatter([center], [center], marker="+", s=200, color="white", linewidths=2, label="frame center (star)")
ax.scatter([col_expected], [row_expected], s=260, facecolors="none", edgecolors=pal.cyan,
           linewidths=2, label=f"expected core, column {col_expected:.1f}")
ax.annotate("increasing column = $+x$", (4, 4), color="white", fontsize=10)
ax.annotate("increasing row = $+y$", (4, 12), color="white", fontsize=10)
ax.set_xlabel("column")
ax.set_ylabel("row")
ax.legend(loc="upper right", fontsize=8)
plt.show()
```

The frame size is a choice with one soft requirement. The farthest stored
starter node is at radius $\sqrt{9.88^2+11.4^2}=15.08\,\lambda_0/D$ and the
frame half-width is $64\times0.25=16$, so every planet core falls inside the
frame. A smaller frame is allowed, because the parser fixes no minimum, but
sources whose cores fall outside it must still return their in-frame wings.
Choose $N\ge2R_{\rm node}/\text{PIXSCALE}$ with $R_{\rm node}$ the farthest
node radius if downstream use needs every core. For the starter that is
$N\ge120.6$, so the 128 pixels satisfy it and any $N$ from 121 up would.

### How do you test the position of the star against the sky?

The checks inside the delivery cannot see a global error in the frame, so add a
fiducial test of position. For a source at $(s,0)$ the centroid of the core,
taken over pixels within 1 $\lambda_0/D$ of the expected position, must sit at
column $c+s/\text{PIXSCALE}$ and row $c$, and a source at $(0,s)$ must sit at
row $c+s/\text{PIXSCALE}$. The second source catches a swapped $x$ and $y$. Use
an $s$ larger than the occulter radius so that a core exists to measure. The
cell runs the test on the toy.

```{code-cell} python
def core_centroid(image, source):
    """Centroid of the pixels within 1 lambda0/D of the expected core position."""
    image = np.asarray(image)
    rows, cols = np.indices(image.shape)
    col0 = center + source[0] / toy.pixscale_lod
    row0 = center + source[1] / toy.pixscale_lod
    near = (cols - col0) ** 2 + (rows - row0) ** 2 <= (1.0 / toy.pixscale_lod) ** 2
    weight = image * near
    return (weight * cols).sum() / weight.sum(), (weight * rows).sum() / weight.sum()

s = 3.0
worst = 0.0
for source in ((s, 0.0), (0.0, s)):
    col, row = core_centroid(toy.psf(jnp.array(source)), source)
    col0, row0 = center + source[0] / toy.pixscale_lod, center + source[1] / toy.pixscale_lod
    worst = max(worst, abs(col - col0), abs(row - row0))
    print(f"source {source}: expected column {col0:.2f}, row {row0:.2f}; "
          f"measured column {col:.2f}, row {row:.2f}")
glue("centroid_err", f"{worst:.1f}", display=False)
```

On the toy the measured centroids land within {glue:text}`centroid_err` pixel of
the expected positions, which is the agreement the delivered cube must show.

A quadrant also needs the center to be exactly where the format says, and yippy.sinc
checks that on admission. The cell moves the declared center of the toy by half a
pixel and admits the grid.

```{code-cell} python
good = tc.quadrant_delivery(toy)
pixels = good["products"].pixels
from dataclasses import replace
moved = replace(good["products"], pixels=replace(
    pixels, center_x=pixels.center_x + 0.5, center_y=pixels.center_y + 0.5
))
admission = sc.admit_grid(**dict(good, products=moved))
print("center moved by half a pixel:", sc.reason_names(admission.reason_bits))
```

The refusal is `FRAME_UNSUPPORTED`. It catches a declared center that is off, but
not images that were actually propagated with the star off center, which is why
the fiducial test above is separate.

### Does a consistent error in the frame get caught?

To answer, first see what the mirror checks compare. A mirror check is an image
propagated directly at a negative position, such as $(-x, y)$. The reader
predicts it from the stored image at $(x, y)$ by reversing the order of its
columns, and admission compares the two ({doc}`../theory/quadrant-symmetry`). The
other two checks reverse the rows, or both. The figure shows the three
reversals on one toy image, for a source at $(3.04, 2.28)$, which is off both
axes so that every reversal moves the core. Look at where the bright core goes:
to the other side of the center when the columns are reversed, below it when
the rows are reversed, and to the opposite corner when both are, which is a
rotation by 180 degrees.

```{code-cell} python
:tags: [hide-input]
node_image = np.asarray(toy.psf(jnp.array([3.04, 2.28])))  # off both axes, so each flip moves the core
panels = {
    "as propagated": node_image,
    "columns reversed ($x$ flipped)": node_image[:, ::-1],
    "rows reversed ($y$ flipped)": node_image[::-1, :],
    "both reversed (rotated 180 degrees)": node_image[::-1, ::-1],
}
fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), constrained_layout=True)
for ax, (name, panel) in zip(axes, panels.items()):
    ax.imshow(np.log10(panel + 1e-12), origin="lower", cmap="magma", vmin=-9, vmax=0)
    ax.scatter([center], [center], marker="+", s=120, color="white", linewidths=1.5)
    ax.set_title(name, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
```

```{admonition} Predict before reading on
Suppose a delivery has a consistent sign error in its detector axes, so that
every image, including the interior checks and the mirror checks, is rotated
by 180 degrees, as in the last panel. Will yippy.sinc admit it? Commit to yes or
no, then run the cell and open the answer.
```

The cell rotates every image of the toy delivery, nodes and mirror checks
together, and admits both versions. It also prints the brightest column of the
core of the $(3.04, 0)$ node, which is the fiducial test in one line.

```{code-cell} python
:tags: [hide-input]
rotated = dict(good)
rotated["intensity_nodes"] = good["intensity_nodes"][..., ::-1, ::-1]
mirrors = good["mirror_checks"]
rotated["mirror_checks"] = eqx.tree_at(
    lambda m: m.intensity, mirrors, mirrors.intensity[..., ::-1, ::-1])
for name, delivery in (("as propagated", good), ("rotated by 180 degrees", rotated)):
    admission = sc.admit_grid(**delivery)
    print(f"{name}: admitted={admission.admitted}, "
          f"mirror residuals {', '.join(f'{v:.1e}' for v in admission.receipt.mirror_local_peak)}")
node = 8  # training row of the node at (3.04, 0): index = iy * 31 + ix with ix = 8
peaks = {}
for name, delivery in (("as propagated", good), ("rotated by 180 degrees", rotated)):
    peaks[name] = int(np.argmax(np.asarray(delivery["intensity_nodes"][0, node]).max(axis=0)))
    print(f"core of the (3.04, 0) node, {name}: brightest column {peaks[name]}")
expected = center + 3.04 / toy.pixscale_lod
starter_expected = (128 - 1) / 2 + 3.04 / 0.25
print(f"expected column c + s / PIXSCALE = {expected:.2f} on the toy's {toy.n_pix} pixels "
      f"({starter_expected:.2f} on the starter's 128)")
glue("peak_ok", str(peaks["as propagated"]), display=False)
glue("peak_rot", str(peaks["rotated by 180 degrees"]), display=False)
glue("expected_toy", f"{expected:.2f}", display=False)
```

```{dropdown} Answer to the prediction
Yes, both versions are admitted, with mirror residuals of order $10^{-14}$ to
$10^{-16}$ in both cases, so no check in the receipt separates them. The
consistent 180 degree rotation commutes with both reversals, so a rotated
mirror check still compares like with like, and the interior reads still agree
with their rotated direct images.

Admission tests that the images agree with each other, and a consistent error
agrees with itself. A delivery is correct when it passes that self-check and
also places the star where the sky says. The fiducial test does: on the toy's
96 pixels the core of the $(3.04,0)$ node is at column {glue:text}`peak_ok`,
next to the expected {glue:text}`expected_toy`, as propagated, and at column
{glue:text}`peak_rot`, on the opposite side of the frame, after the rotation.
Run the fiducial test on the delivered files before sending.
```

(recipe-pixel)=
## Step 3: What does one pixel hold?

A pixel value can mean three different things, and the library needs to be
told which. A **pixel integral** is the intensity collected over the area of the
pixel, a mean density is that integral divided by the pixel area, and a point
sample is the intensity at the pixel center alone. The figure shows one toy
planet core, cut along the row through its center, at the toy's own 0.25
$\lambda_0/D$ pixels. Look at how the point samples (dots) and the pixel means
(steps) sit on the underlying curve, and at the right panel, which plots their
difference as a fraction of the peak.

```{code-cell} python
:tags: [hide-input]
source = jnp.array([3.0, 0.0])
binned, fine_image = {}, None
for factor in (1, 2, 4, 8):
    fine = tc.ToyCoronagraph(n_pix=toy.n_pix * factor, pixscale_lod=toy.pixscale_lod / factor)
    image = np.asarray(fine.psf(source))
    if factor == 8:
        fine_image, fine_x = image, np.asarray(fine.detector_x)
    binned[factor] = image.reshape(toy.n_pix, factor, toy.n_pix, factor).mean(axis=(1, 3))
row = toy.n_pix // 2
columns = slice(row + 4, row + 28)
x = np.asarray(toy.detector_x)[columns]
point = binned[1][row, columns]
mean = binned[8][row, columns]
fine_row = fine_image[8 * row + 4]
fine_keep = (fine_x >= x[0] - 0.125) & (fine_x <= x[-1] + 0.125)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
axes[0].plot(fine_x[fine_keep], fine_row[fine_keep], color="0.6", label="underlying intensity")
axes[0].bar(x, mean, width=toy.pixscale_lod, alpha=0.35, color=pal.pink, label="pixel mean")
axes[0].scatter(x, point, color=pal.cyan, zorder=3, s=18, label="point sample")
axes[0].set_xlabel(r"$x$ [$\lambda_0/D$]")
axes[0].set_ylabel("intensity / unocculted peak")
axes[0].legend(fontsize=8)
axes[1].plot(x, 100 * (point - mean) / mean.max(), color=pal.yellow, marker="o")
axes[1].axhline(0, color="0.5", linewidth=0.8)
axes[1].set_xlabel(r"$x$ [$\lambda_0/D$]")
axes[1].set_ylabel("point minus mean [% of peak]")
plt.show()
```

The pixel integral is the pixel mean times the pixel area, so the integral and
the mean carry the same information, in different units. The point sample is
a different number, which is off from the mean by
{glue:text}`gap_pct` percent of the peak on this core, and the difference cannot be
repaired after the fact. The toy images of these pages are point samples, so
they exercise admission and not the pixel-integral contract that this step
recommends for a real delivery.

For a pixel-integrated delivery, normalize to the photons that
enter the aperture: each stored value is the fraction of incident photons that
land on that pixel (`sample_kind: integral`, `normalization.kind:
incident_photon`, [pixel measure][schema-pixel] and [normalization][schema-norm]).
The Fourier normalization can be checked with Parseval's relation.

:::{admonition} Known result: Parseval's relation
:class: note

The intensity of the Fourier transform of a field, integrated over the whole
focal plane, equals the intensity of the field integrated over the pupil (Rayleigh's
energy theorem; Goodman, *Introduction to Fourier Optics*, chapter 2, Fourier
transform theorems). The cell below checks the discrete version for the toy's
pupil.
:::

```{code-cell} python
n_pupil, n_fft = toy.n_pupil, 512
pupil = np.asarray(toy.pupil)
focal = np.fft.fft2(pupil, s=(n_fft, n_fft))
focal_energy = (np.abs(focal) ** 2).sum() / n_fft**2
pupil_energy = (pupil**2).sum()
print(f"focal-plane energy / pupil energy = {focal_energy / pupil_energy:.12f}")
print(f"pupil area = {pupil.sum() / n_pupil**2:.4f} D^2 (pi / 4 = {np.pi / 4:.4f})")
```

The intensity over the whole focal plane therefore integrates to the pupil area
in units of $D^2$, and the fraction in a pixel is the intensity integrated over
the pixel, in $(\lambda_0/D)^2$, divided by that pupil area.

Most propagators return point samples at pixel centers, so integrate in the
propagator by supersampling. Propagate on a grid $s$ times finer in each axis
and average each $s\times s$ block, which is the pixel mean, then multiply by the
pixel area for the integral. The cell measures how fast this converges for the
planet core of the figure, with $s=8$ as the reference.

```{code-cell} python
gaps = {}
for factor in (1, 2, 4):
    gaps[factor] = np.abs(binned[factor] - binned[8]).max() / binned[8].max()
    print(f"supersampling {factor}: largest pixel difference from s=8, "
          f"as a fraction of the peak: {gaps[factor]:.1e}")
print(f"ratios: s=1 to 2 {gaps[1] / gaps[2]:.1f}, s=2 to 4 {gaps[2] / gaps[4]:.1f}")
glue("gap_pct", f"{100 * gaps[1]:.1f}", display=False)
glue("gap_s4", tex(float(gaps[4])), display=False)
glue("ratio_12", f"{gaps[1] / gaps[2]:.1f}", display=False)
glue("ratio_24", f"{gaps[2] / gaps[4]:.1f}", display=False)
v = load("validation.json")
glue("level", tex(1e-5, 1), display=False)
glue("starter_worst", tex(v["worst_check_error_local_peak"]), display=False)
```

A point sample at the pixel center is off by {glue:text}`gap_pct` percent of the
peak on this core. Doubling $s$ cuts the difference by a factor of
{glue:text}`ratio_12` and then {glue:text}`ratio_24`, reaching {glue}`gap_s4` at
$s=4$. Choose $s$ so that the difference is well below the interior acceptance
level of {glue}`level` of the local peak. The level is recommended in
{doc}`recipe-check-send` because the clean starter's worst interior check reads
{glue}`starter_worst`, and a toy core like this one would need an $s$ above 4 to
get there. Confirm the choice with the factor-two refinement that step 6 asks
for.

If the propagator reports an unocculted-peak image instead, the other
accepted reference is `unocculted_peak`, with a positive `photon_conversion`
per plane that is the same for every source position. A conversion that
varies with position breaks the band limit
({doc}`../theory/band-limits`). Never normalize an image to its own sum or
peak. For a point-sampled delivery, store the centre density times the cell area and
declare `sample_kind: point`. The reader recovers the centre density.

### What does the diameter convention change?

The unit $\lambda_0/D$ and the width $L$ are both measured against $D$, so the
declared `diameter_convention` and the declared support must agree, and the
choice has to be made before step 1. The figure draws a regular hexagonal
aperture. Look at the two widths marked: the flat-to-flat width $F$ across the
flats, and the vertex-to-vertex width across the corners, which is
$2F/\sqrt3\approx1.155F$.

```{code-cell} python
:tags: [hide-input]
angles = np.deg2rad(np.arange(0, 360, 60))
corners = np.stack([np.cos(angles), np.sin(angles)], axis=1)
fig, ax = plt.subplots(figsize=(4.4, 3.8), constrained_layout=True)
ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=pal.cyan, linewidth=2))
flat = np.sqrt(3) / 2
ax.annotate("", (-1, -1.18), (1, -1.18), arrowprops=dict(arrowstyle="<->", color="white"))
ax.text(0, -1.38, r"vertex to vertex $=2F/\sqrt{3}$", ha="center", color="white")
ax.annotate("", (1.25, -flat), (1.25, flat), arrowprops=dict(arrowstyle="<->", color="white"))
ax.text(1.32, 0, r"flat to flat $F$", rotation=90, va="center", color="white")
ax.set_xlim(-1.3, 1.7)
ax.set_ylim(-1.6, 1.1)
ax.set_aspect("equal")
ax.axis("off")
plt.show()
```

Take that aperture. With $D$ the circumscribed width, the vertex-to-vertex
width, $L=1$ and the spacing is 0.38. With $D=F$, inscribed, the illuminated
width along the vertex axis is $L=2/\sqrt3$ and the spacing is $0.76/(2L)$ at
$\lambda_0$.

```{code-cell} python
L_inscribed = 2 / np.sqrt(3)
h_inscribed = 0.76 / (2 * L_inscribed)
print(f"L = {L_inscribed:.3f}; spacing = 0.76 / (2 * L) = {h_inscribed:.3f} lambda0/F")
print(f"in units of lambda0 over the circumscribed width: {h_inscribed / (np.sqrt(3) / 2):.3f}")
glue("hex_L", f"{L_inscribed:.3f}", display=False)
glue("hex_h", f"{h_inscribed:.3f}", display=False)
glue("hex_h_circ", f"{h_inscribed / (np.sqrt(3) / 2):.3f}", display=False)
```

That is $L=$ {glue:text}`hex_L` and a spacing of {glue:text}`hex_h` $\lambda_0/F$.
Since $F=(\sqrt3/2)$ times the circumscribed width, it is the same physical
spacing, {glue:text}`hex_h_circ` $\lambda_0$ per circumscribed width, and only the unit
changed. Either convention is acceptable if it is stated and used for both $D$
and $L$.

## What this page established

Every image sits in one frame with the star at $((n_x-1)/2,\,(n_y-1)/2)$, and
a fiducial source at a known position must land where the sky puts it, because
admission tests that the images agree with each other and a consistent error
agrees with itself. The pixel response and sampling declaration determine whether to
store a centre sample, a mean or an integral. Supersampling in the propagator
approximates the latter two for a uniform square pixel. The next page handles the filter and the
symmetry argument.

Next: {doc}`recipe-band-symmetry`.

[schema-fits]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#header-keywords
[schema-pixel]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#pixel-measure
[schema-norm]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#normalization
