---
file_format: mystnb
kernelspec:
  name: python3
---

# How wrong is a read between the nodes?

A planet is rarely where a designer propagated an image, so every use of a
delivered grid starts with a read at a position nobody computed. The number
that comes back is only worth using if the grid was fit for the job and the
read can be checked. This tutorial admits a grid of PSFs, reads it between
its nodes and checks the reads against direct propagation. You end with a
measured error, not a certified one, and with the list of things the
library did and did not promise.

:::{admonition} The question of this page
:class: important

How far is a read between the nodes from direct propagation, where is it
worst, and what does the library claim about it? Each section answers one
part, and the close fills in the answers.
:::

```{code-cell} python
:tags: [remove-cell]
import eyepiece as ep
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown
from myst_nb import glue

hwostyle.use("dark")
pal = hwostyle.palette


def put(name, text):
    """Glue a Markdown string under a name the prose can cite."""
    glue(name, Markdown(text), display=False)


def sci(value, digits=1):
    """Format a number as inline math, a times a power of ten."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"
```

## What is delivered, and what is a read?

The grid comes from the toy Lyot coronagraph of the theory pages, whose
helper module (`toy_coronagraph.py`, in the `docs` folder of the repository)
builds a complete quadrant grid and the declarations yippy.sinc requires.
{doc}`declarations` explains each declaration, and here they are taken as
given. This toy grid is not the public delivery that {doc}`yippy` and
{doc}`../validation` use, which has 27 by 31 nodes.

The first cell enables JAX 64-bit mode, because yippy.sinc refuses to build a
reader without it, and builds the delivery. The toy's numbers are in the box
that follows.

```{code-cell} python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import yippy.sinc as sc
import toy_coronagraph as tc

toy = tc.ToyCoronagraph()
delivery = tc.quadrant_delivery(toy)
```

```{code-cell} python
:tags: [remove-cell]
from yippy.sinc.geometry import origin_xy, spacing_xy
products = delivery["products"]
lattice = delivery["declared_lattice"]
step = float(spacing_xy(lattice)[0])
count = int(lattice.shape[0])
half = float(delivery["domain"].limits_xy_lod[1, 0])
rho = 2 * step * float(products.support_extent_xy[0, 0])
put("t_npix", f"{toy.n_pix} by {toy.n_pix}")
put("t_pitch", f"{toy.pixscale_lod:g}")
put("t_field", f"{toy.n_pix * toy.pixscale_lod:g}")
put("t_occ", f"{toy.occulter_lod:g}")
put("t_lam", f"{float(products.frame.reference_wavelength_nm):g}")
put("t_dia", f"{float(products.frame.diameter_m):g}")
put("t_lyot", f"{toy.lyot:g}")
put("t_step", f"{step:g}")
put("t_nodes", f"{count} by {count}")
put("t_rho", f"{rho:.2f}")
put("t_half", f"{half:g}")
put("t_served", f"{step * (count - 16):.1f}")
put("t_extent", f"{step * (count - 1):.1f}")
```

(toy-in-numbers)=
## The toy in numbers

:::{admonition} The toy in numbers
:class: note

- **Detector.** {glue}`t_npix` pixels, {glue}`t_pitch` $\lambda_0/D$ per pixel, so the
  image spans {glue}`t_field` $\lambda_0/D$ on a side.
- **Optics.** Reference wavelength $\lambda_0=$ {glue}`t_lam` nm, entrance diameter
  $D=$ {glue}`t_dia` m, Lyot stop {glue}`t_lyot` of the diameter, and a hard occulting disk of
  radius {glue}`t_occ` $\lambda_0/D$. A planet closer than that to the star sits
  behind the disk and is dimmed.
- **Grid.** {glue}`t_nodes` nodes in the positive quadrant at spacing {glue}`t_step`
  $\lambda_0/D$. The entrance pupil is one diameter wide, so the oversampling
  factor is $\rho=2Bh=$ {glue}`t_rho`, the kernel margin of the
  {doc}`kernel page <../theory/kernel>`. That value is chosen to show the
  grid exactly at the margin, and {doc}`admission` shows what happens at coarser spacing.
- **Permission box.** Reads are allowed for positions inside a box of
  half-width {glue}`t_half` $\lambda_0/D$.
- **Point samples.** The toy images are point samples of intensity at
  pixel centers, not pixel integrals. They therefore exercise admission, but
  they are not the format that the delivery recipe recommends for a real
  delivery ({ref}`recipe-pixel`).
:::

## Where are the nodes, the permission box and one read's stencil?

The next figure shows the delivered nodes, the permission box and the
**stencil** of one read at $(4.0, 0.3)$. A node is a source position for
which an image was propagated. The grid stores only the positive quadrant
(cyan points), and the figure reflects it across both axes, as the
{doc}`quadrant symmetry page <../theory/quadrant-symmetry>` allows, so the
hollow gray points are mirrored copies. The stencil is the 32 by 32 block of
nodes that one read combines, as taught on the
{doc}`sinc page <../foundations/sinc-interpolation>`.

```{code-cell} python
:tags: [hide-input]
from yippy.sinc import viz

query = np.array([4.0, 0.3])
fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
viz.plot_coverage(lattice, domain=delivery["domain"], quadrant=True,
                  query=tuple(query), ax=ax)
plt.show()
```

Look at the white cross, the query, and the large white rectangle around it,
the stencil: it lies inside the nodes. The inner outline is the permission
box, and the shaded rectangle is every position whose stencil fits inside
the nodes. A read is served only inside both. A read near the edge of the
nodes has a stencil that runs off them, and the last section shows what
happens then.

## Does admission accept the grid?

Admission is a one-time check of the whole grid, and it either returns a
usable grid or explains why not.

```{code-cell} python
admission = sc.admit_grid(**delivery)
print("admitted:", admission.admitted)
print("reasons:", sc.reason_names(admission.reason_bits))
print("receipt:", admission.receipt.identity[:16], "...")
```

The grid is admitted and no reason bits are set. The receipt is the small
record of what admission checked and decided: its identity is a hash of the
data, the declarations and their provenance, so readmitting changed data
gives a new identity.

## What does one read return?

An `Interpolator` binds the admitted grid to a permission region, the kernel
profile, a `ReadPolicy` (which says whether a read must carry a certified
error bound), a `Tiling` (how many detector pixels one read handles at a
time, which sets its scratch memory) and a memory budget (the byte ceilings
that reads and admission may not exceed).

```{code-cell} python
reader = sc.Interpolator(
    admission.require(),
    delivery["domain"],
    sc.KernelProfile(),
    sc.ReadPolicy(),
    sc.Tiling(pixel_tile=1024),
    delivery["memory"],
)
result = reader.read(jnp.array([4.0, 0.3]), jnp.int32(0))
print("accepted:", bool(result.accepted), " shape:", result.intensity.shape)
print("error bound:", float(result.abs_error_bound))
print("reasons:", sc.reason_names(int(result.reason_bits)))
```

The call `reader.read(position, product)` takes a source position and a
product index, the place of the wavelength plane in the delivery's list of
products, so index 0 is the first and here the only one. The read returns the
image together with its status. The default policy is sampling mode: a read
returns its image whenever the grid was admitted and the position is
permitted. It attaches no error bound, so the bound prints as `inf`, which
means unknown and not large.

One reason bit appears even on this accepted read. `BOUND_UNKNOWN` is set on
every read in version 0.1, accepted or not, because no certified bound exists
yet ({doc}`admission` explains why). It is a standing note that the read is
measured rather than certified. The flag to act on is `accepted`. The
comparison below is how to judge the read.

## How far is the read from direct propagation?

The toy can propagate the planet directly at the same position, which gives
the truth. The residual is shown relative to the **local peak**, the
brightest direct pixel within 1 $\lambda_0/D$ of the planet, and that is the
only meaning of "the planet's own peak" on this page.

```{code-cell} python
:tags: [hide-input]
px = np.asarray(toy.detector_x)
xx, yy = np.meshgrid(px, px)


def local_peak(image, source):
    """Brightest pixel of an image within 1 lambda0/D of the source position."""
    near = (xx - source[0]) ** 2 + (yy - source[1]) ** 2 <= 1.0
    return image[near].max()


truth = np.asarray(toy.psf(query))
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
viz.plot_read(result, pixels=products.pixels, ax=a, vmin=1e-9, vmax=0.5)
res = viz.plot_residual(result, truth, source_xy_lod=query,
                        pixels=products.pixels, scale="magnitude", ax=b)
for ax in (a, b):
    ax.set_xlim(-1, 8)
    ax.set_ylim(-4.5, 4.5)
a.set_title("yippy.sinc read at (4.0, 0.3)")
b.set_title("|read - direct| / local peak")
plt.show()
one_error = np.abs(np.asarray(result.intensity) - truth).max() / local_peak(truth, query)
floor = tc.kernel_error_worst(rho)
put("c_one", sci(one_error))
put("c_floor", sci(floor))
print(f"worst residual / local peak: {one_error:.1e}; "
      f"kernel worst-tone error at rho = {rho:.2f}: {floor:.1e}")
```

Look at the circle in the right panel, which marks the disk where the local
peak is taken: the residual has no structure tied to the planet's core. The residual is at worst {glue}`c_one` of the local peak, spread
over the image, which is the kernel's truncation error. The kernel page
measured a worst-tone error of {glue}`c_floor` for a unit-amplitude tone at
$\rho=$ {glue}`t_rho`. Both numbers are an error divided by a signal size, but the
denominators differ (a tone's amplitude there, the planet's local peak here),
so the comparison says the read is no worse than the floor and proves nothing
about other positions.

## Where is the error largest?

`read_batch` maps a batch of positions through the reader in fixed-size
chunks, and the reader works inside `eqx.filter_jit`. The cell reads 60
random positions between 0.5 and 4.8 $\lambda_0/D$ from the star.

```{code-cell} python
import equinox as eqx

rng = np.random.default_rng(1)
radius = rng.uniform(0.5, 4.8, 60)
angle = rng.uniform(0, 2 * np.pi, 60)
positions = jnp.asarray(np.stack([radius * np.cos(angle), radius * np.sin(angle)], 1))
batch = eqx.filter_jit(sc.read_batch)(reader, positions, jnp.zeros(60, dtype=jnp.int32))
print("all accepted:", bool(batch.accepted.all()), " images:", batch.intensity.shape)
```

An error can be measured against two references. One is the local peak used
so far. The other is each pixel's brightest value over all stored source
positions, the most light that pixel ever receives as the planet moves. The
grid stores only the positive quadrant, so the brightest-value map takes the
maximum over the stored images and their three mirror images, which covers
planets in all four quadrants.

```{admonition} Predict before reading on
:class: tip
At $(4.0, 0.3)$, outside the occulter, the read was within {glue}`c_one` of the
local peak. A planet at about 1.7 $\lambda_0/D$ sits behind the occulting
disk, whose radius is {glue}`t_occ` $\lambda_0/D$, and its image is much
fainter than outside. Will the error relative to the local peak be smaller,
about the same, or larger?
```

```{dropdown} Answer
Larger. The absolute error in a pixel follows how bright that pixel gets
for the planet positions in its stencil, not how bright the occulted
planet's own faint image is. The absolute error stays about the same while
the local peak, the denominator, falls. The figure below shows the numbers.
```

```{code-cell} python
:tags: [hide-input]
truths = np.asarray(toy.psf(positions))
error = np.abs(np.asarray(batch.intensity) - truths)
sources = np.asarray(positions)
peaks = np.array([local_peak(t, s) for t, s in zip(truths, sources)])
stored = np.asarray(delivery["intensity_nodes"][0])
brightest = stored.max(axis=0)
brightest = np.maximum.reduce([brightest, brightest[:, ::-1], brightest[::-1, :],
                               brightest[::-1, ::-1]])
relative_local = error.max(axis=(1, 2)) / peaks
relative_pixel = (error / brightest).max(axis=(1, 2))
behind = radius < toy.occulter_lod

fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
ax.semilogy(radius, relative_local, "o", color=pal.pink, ms=4)
ax.semilogy(radius, relative_pixel, "s", color=pal.cyan, ms=4)
ax.axvline(toy.occulter_lod, color=ep.style.neutral(0.4), lw=1, ls="--")
ax.text(0.55, relative_local.max() * 1.6, "relative to the local peak",
        color=pal.pink, ha="left", va="bottom", fontsize=9)
ax.text(4.8, relative_pixel.max() * 1.8, "relative to each pixel's brightest value",
        color=pal.cyan, ha="right", va="bottom", fontsize=9)
ax.set_ylim(relative_pixel.min() * 0.5, relative_local.max() * 5)
ax.text(toy.occulter_lod + 0.05, relative_pixel.max() * 30, "occulter edge", ha="left",
        va="center", fontsize=9, color=ep.style.neutral(0.6))
ax.set_xlabel(r"planet radius ($\lambda_0/D$)")
ax.set_ylabel("max |read - direct|")
plt.show()
put("b_behind", sci(relative_local[behind].max()))
put("b_outside", sci(relative_local[~behind].max()))
put("b_pixel", sci(relative_pixel.max()))
put("b_n", str(int(behind.sum())))
put("b_peak", sci(np.median(peaks[behind])))
put("b_all", str(int(np.asarray(batch.accepted).sum())))
print(f"worst relative to local peak: behind the disk {relative_local[behind].max():.1e}, "
      f"outside {relative_local[~behind].max():.1e}; "
      f"worst relative to pixel brightest: {relative_pixel.max():.1e}")
```

The two series are the same errors divided by two references. Look at the
pink series, relative to the local peak: it rises behind the occulter, up to
{glue}`b_behind` for the {glue}`b_n` sampled positions inside the disk, against {glue}`b_outside` outside it. The
median local peak behind the disk is {glue}`b_peak` of the unocculted peak, so the denominator
falls while the error does not. Relative to each pixel's brightest value
(cyan) the error is flat, at most {glue}`b_pixel` at every sampled radius.

For science that cares about an occulted planet's photometry, the absolute
error is what matters, and it is the same small number everywhere. That
{glue}`b_pixel` is an error over the most light the pixel ever receives. It
is the same kind of ratio as the kernel's {glue}`c_floor`, and below it, but
it is not a bound at positions that were not sampled.

## What does a refusal look like?

A read outside the permission region, or too close to the edge of the nodes,
is refused. Compiled reads return a refused status and a NaN image instead of
raising, so a batch never stops halfway, and `require_read` is the eager helper
that raises with the refused indices and reasons. The cell reads two
positions on the $x$ axis, one between the permission box (half-width
{glue}`t_half`) and the last complete stencil ({glue}`t_served`), and one beyond both.

```{code-cell} python
for x in (5.3, 6.5):
    outside = reader.read(jnp.array([x, 0.0]), jnp.int32(0))
    print(f"x = {x}: accepted:", bool(outside.accepted),
          " status:", sc.Status(int(outside.status)).name,
          " image is NaN:", bool(np.isnan(np.asarray(outside.intensity)).all()))
    print("  reasons:", sc.reason_names(int(outside.reason_bits)))
try:
    sc.require_read(outside)
except sc.ReadRefused as err:
    print("ReadRefused:", type(err).__name__)
```

Each refused read lists its reasons. `OUTSIDE_DOMAIN` means the position is
outside the permission box, so both positions carry it.
`INCOMPLETE_STENCIL` means the 32 by 32 block of nodes around the position
would reach past the stored grid. The grid stores nodes out to
{glue}`t_extent` $\lambda_0/D$, but a stencil needs 16 nodes on each side, so
the last position with a complete stencil is {glue}`t_served`. The position at
5.3 is outside the box and still has a complete stencil. The one at 6.5 has
neither, as the next figure shows with the stencil running off the nodes.

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
viz.plot_coverage(lattice, domain=delivery["domain"], quadrant=True,
                  query=(6.5, 0.0), ax=ax)
plt.show()
```

## What did this tutorial establish?

```{code-cell} python
:tags: [remove-cell]
put("s_id", admission.receipt.identity[:12])
```

The question was how far a read between the nodes is from direct
propagation, where it is worst, and what the library claims. The answers,
filled in:

- **Admitted:** yes, with no reason bits and receipt identity {glue}`s_id`.
- **Reads accepted:** {glue}`b_all` of 60 sampled positions inside the box.
- **Error against direct propagation:** at most {glue}`b_outside` of the
  local peak outside the occulter and {glue}`b_behind` behind it.
- **Where worst:** behind the occulter, because the local peak is small there.
  The absolute error is about the same everywhere, at most {glue}`b_pixel`
  of each pixel's brightest value.
- **Claimed by the library:** nothing certified. The error bound is `inf`,
  `BOUND_UNKNOWN` is set on every read, and the numbers above are measurements
  at 60 positions.
- **Refused:** reads outside the box ({glue}`t_half`) or past the last complete
  stencil ({glue}`t_served`), with a named reason each.

The next page shows how to read a delivery through yippy, and
{doc}`admission` explains what admission refuses and how to fix it.
{doc}`prefiltered-read` trades some of this accuracy for a read that sums 49
images instead of 1,024.

Next: {doc}`yippy`.
