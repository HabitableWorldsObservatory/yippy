---
file_format: mystnb
kernelspec:
  name: python3
---

# Can a read use fewer images?

The read of {doc}`first-read` sums a 32 by 32 block of delivered images, 1,024
of them for every position. That is what makes it slower than a blend, and a
fit or a yield run asks for many positions. This page reads the same kind of
grid from a 7 by 7 block, 49 images, by doing part of the work once when the
grid is loaded. It shows what that costs in accuracy, what the grid must
satisfy, and how to render a scene of many sources in one pass.

:::{admonition} The question of this page
:class: important

What does a read from 49 images instead of 1,024 give up, what does the
library record about it, and when is it refused?
:::

```{code-cell} python
:tags: [remove-cell]
import numpy as np
from IPython.display import Markdown
from myst_nb import glue


def put(name, text):
    """Glue a Markdown string under a name the prose can cite."""
    glue(name, Markdown(text), display=False)


def sci(value, digits=1):
    """Format a number as inline math, a times a power of ten."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"
```

## What is done once, and what is done per read?

The sinc weights of the raw read fall off slowly, which is why it needs 16
nodes on each side of the query. A short, smooth bump falls off fast, so a
read that weights nodes with a bump needs only a few of them. A bump is the
wrong interpolation weight for the delivered samples, though: it blurs them.

The fix is an old one from the non-uniform fast Fourier transform (Dutt and
Rokhlin 1993; Fessler and Sutton 2003; Barnett, Magland and af Klinteberg
2019). Filter the samples once so that the blur of the bump is undone in
advance, store the filtered values, and read those with the bump.

1. **At load.** Each delivered image stack is filtered along both source axes.
   The results are the **coefficient images**, one for each node, held beside
   the raw samples. The delivery file is unchanged: it still holds raw samples.
2. **Per read.** A read takes the `taps` nearest coefficient images along each
   axis and sums them with Kaiser-Bessel bump weights. With 7 taps that is 49
   images.

The bump for `taps` nodes weights a node at signed distance $u$ node spacings
from the query by

$$
b(u)=\frac{I_0\!\left(\beta\sqrt{1-(2u/\mathrm{taps})^2}\right)}{I_0(\beta)}
\quad\text{for } |u|<\mathrm{taps}/2,\qquad b(u)=0 \text{ otherwise},
$$

where $I_0$ is the modified Bessel function of order zero. The filter is the
least-squares filter of smallest norm that makes the filtered-then-bumped
read reproduce every cosine inside the declared band, and the shape parameter
$\beta$ is chosen for the band. Both are computed in float64 when the grid is
admitted. The design band is the one at the shortest declared wavelength,
where the image varies fastest with source position.

## Which tap count meets a tolerance?

The tap count is a setting of the load, not of the file. `plan_prefilter`
picks it from the delivery's metadata alone: it tries 5 to 10 taps in order
and returns the first whose **tone error** is at or below the request.

The tone error is the largest error of the read on a cosine of unit amplitude
inside the band, searched over a grid of frequencies and query positions and
combined over the two axes. It is a sampled maximum for the kernel, not an
error bound on a delivered image.

The grid is the toy quadrant of {doc}`first-read`, delivered at a spacing of
0.32 $\lambda_0/D$ instead of 0.38, for a reason the admission section gives.

```{code-cell} python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import yippy.sinc as sc
from yippy.sinc.evidence import reason_names
from yippy.sinc.kernel import PrefilteredKernelProfile
from yippy.sinc.planner import plan_prefilter
from yippy.sinc.reader import PrefilteredReader, read_batch
import toy_coronagraph as tc

toy = tc.ToyCoronagraph()
delivery = tc.quadrant_delivery(toy, spacing=0.32, nodes=31)
metadata = {
    name: delivery[name]
    for name in ("declared_lattice", "products", "reflection", "domain")
}
plan = plan_prefilter(**metadata, max_tone_error=5e-5)
print("planned:", plan.planned, " taps:", plan.profile.taps)
for taps, error in plan.candidates:
    print(f"  {taps} taps: tone error {error:.1e}")
```

In this list each extra tap buys a factor of six to eight. The planner stopped at
the first count that met $5\times10^{-5}$. A request that 10 taps cannot meet
is refused with a reason and with the list of what was tried:

```{code-cell} python
strict = plan_prefilter(**metadata, max_tone_error=1e-8)
print("planned:", strict.planned, reason_names(strict.reason_bits))
print("best tried:", f"{strict.candidates[-1][0]} taps, {strict.candidates[-1][1]:.1e}")
```

`plan.require()` returns the profile, or raises `PlanRefused` carrying the
plan. A profile can also be built directly, `PrefilteredKernelProfile(taps=7)`,
which is the default.

## Does admission accept the grid for this profile?

Admission takes the profile as a keyword. It designs the filter for the
grid and checks everything the raw profile checks, with three differences.

- **Denser sampling.** The oversampling factor $\rho$ must be at most 0.64 on
  every axis, where the raw profile allows 0.76. A grid delivered at the raw
  limit is refused for the prefiltered profile with `KERNEL_MARGIN_FAIL`,
  which is why this page uses a spacing of 0.32.
- **Smaller grids are enough.** A quadrant needs 13 nodes on each axis for 7
  taps and 17 otherwise; a full grid needs 25 and 33.
- **Independent checks are read with the profile.** Checks at positions
  between the nodes are compared against prefiltered reads, so their residuals
  are evidence for the reader that will be used. Mirror checks compare the
  stored samples themselves, as they do for the raw profile, and do not depend
  on the profile.

```{code-cell} python
profile = plan.require()
admission = sc.admit_grid(**delivery, profile=profile)
print("admitted:", admission.admitted, reason_names(admission.reason_bits))
for name, value in admission.receipt.kernel_design:
    print(f"  {name}: {value:.4g}")
```

The receipt's `kernel_design` is the record of the filter that was built:
tap count, filter half-width in nodes, halo, the design band on each axis in
cycles per node, the bump shape, and the tone errors. `tone_error_two_axis`
is the figure the planner compared with the request. The interior figures
apply to reads far from the table's ends; the larger figures include reads
near an end, where the filter is one-sided. `sample_gain` is the largest sum
of the absolute weights that a read puts on the delivered samples along one
axis: an error of at most $\epsilon$ in every sample can change a read by up
to `sample_gain` times $\epsilon$ for each axis.

The receipt identity includes the profile, so a grid admitted for 7 taps and
the same grid admitted for 8 have different identities.

## What does one read return?

A `PrefilteredReader` takes the same arguments as an `Interpolator`. It builds
the coefficient images when it is constructed.

```{code-cell} python
reader = PrefilteredReader(
    admission.require(),
    delivery["domain"],
    profile,
    sc.ReadPolicy(),
    sc.Tiling(pixel_tile=1024),
    delivery["memory"],
)
query = jnp.array([4.0, 0.3])
result = reader.read(query, jnp.int32(0))
print("accepted:", bool(result.accepted), " shape:", result.intensity.shape)
print("error bound:", float(result.abs_error_bound))
print("reasons:", reason_names(int(result.reason_bits)))
```

The result has the meaning it has for the raw read. The image is returned
because the grid was admitted and the position is served, the bound is
unknown, and `BOUND_UNKNOWN` is set. A policy that demands a bound refuses,
as it does for the raw profile. `read_batch`, `require_read`,
`eqx.filter_jit` and `jax.vmap` work unchanged.

## How far are the reads from direct propagation?

The toy can propagate a planet directly, so the reads can be checked. The
error is divided by the **local peak**, the brightest direct pixel within 1
$\lambda_0/D$ of the planet, as on {doc}`first-read`.

```{code-cell} python
px = np.asarray(toy.detector_x)
xx, yy = np.meshgrid(px, px)


def local_peak_error(image, source):
    """Worst pixel error against direct propagation, over the local peak."""
    truth = np.asarray(toy.psf(jnp.asarray(source)))
    near = (xx - source[0]) ** 2 + (yy - source[1]) ** 2 <= 1.0
    return np.abs(np.asarray(image) - truth).max() / truth[near].max()


rng = np.random.default_rng(3)
positions = jnp.asarray(rng.uniform(-5.0, 5.0, (40, 2)))
batch = read_batch(reader, positions, jnp.zeros(40, dtype=jnp.int32))
errors = [
    local_peak_error(image, np.asarray(position))
    for image, position in zip(batch.intensity, positions)
]
print("all accepted:", bool(jnp.all(batch.accepted)))
print(f"worst error / local peak over 40 reads: {max(errors):.1e}")
```

```{code-cell} python
:tags: [remove-cell]
put("p_worst", sci(max(errors)))
put("p_tone", sci(plan.tone_error_two_axis))
put("p_taps", f"{profile.taps}")
```

The worst of 40 reads is {glue}`p_worst` of the local peak, with
{glue}`p_taps` taps and a design tone error of {glue}`p_tone`. The two
numbers have different denominators (the planet's local peak here, a tone's
amplitude there), and an image is a sum of many tones, so one does not bound
the other. They agree in order of magnitude, which is the most the comparison
can say. Forty positions on a toy are a demonstration, not a qualification.

## Where is a read refused?

A read is served only where its whole stencil lies at least 6 stored nodes
inside the table's ends, because the filter needs samples on both sides of a
node to be accurate there. For a quadrant the mirrored copies supply the
samples across the axes, so only the outer ends lose a margin.

Measured in node spacings from the first node, 7 taps serve positions $t$ with
$8.5\le t<N-9.5$ on a full grid of $N$ nodes, and 8 taps serve
$9\le t<N-10$. On a quadrant the lower limit is the mirror of the upper one.

```{code-cell} python
outside = reader.read(jnp.array([7.5, 0.0]), jnp.int32(0))
print("accepted:", bool(outside.accepted))
print("reasons:", reason_names(int(outside.reason_bits)))
```

This position is beyond both the served stencils and the permission box, and
both reasons are reported.

## What does the prefiltered read give up?

- **A lower spacing limit.** $\rho\le0.64$ instead of $0.76$: about 19 percent
  more nodes on each axis for the same field.
- **A kernel error set by the tap count.** Seven taps are a few times
  $10^{-5}$ at the spacing limit and each further tap gains less than a
  decade. {doc}`../theory/kernel` measures the raw profile's worst-tone error
  for comparison.
- **Small steps where the stencil moves.** The bump is cut at a small nonzero
  value, $1/I_0(\beta)$, so a read changes by a step of about that size,
  relative to the local intensity scale, when the query crosses the point
  where the stencil shifts by one node (about $10^{-6}$ for 7 taps and
  $10^{-7}$ for 8). The step is inside the recorded tone error. The read is
  differentiable in position between those points.
- **The samples are not kept.** The reader holds the coefficient images in
  place of the raw samples, in the same dtype, so its `grid.intensity` is the
  coefficient table. Keep the delivery if the samples are needed again.
- **Coefficient rounding.** The filter runs in float64 and the coefficients
  are stored in the dtype of the samples. For float32 samples that adds a
  rounding of order $10^{-7}$ of the local intensity scale, which the recorded
  tone error does not include.

## How is a scene of many sources rendered?

A disk or a field of planets is a sum of many reads. Because a prefiltered
read is a weighted sum of coefficient images, the sum over sources can be
regrouped by node: each source spreads its flux onto its 49 nodes, and the
accumulated node amplitudes multiply the coefficient images once. The cost
then follows the number of nodes the scene touches, not the number of sources.

`SceneRenderer` fixes that set of nodes when it is built, from positions that
cover where the sources can be.

```{code-cell} python
flux = jnp.asarray(rng.uniform(0.5, 2.0, 40))
renderer = sc.SceneRenderer(reader, 0, positions)
scene = renderer.render_points(positions, flux)
summed = jnp.einsum("q,qrc->rc", flux, batch.intensity)
print("accepted:", bool(scene.accepted), " nodes:", renderer.nodes.shape[0])
print(f"largest difference from the summed reads: "
      f"{float(jnp.max(jnp.abs(scene.intensity - summed))):.1e}")
```

The render equals the flux-weighted sum of the 40 reads to rounding. Like a
single read, it sums in float64 whatever the dtype of the samples, unless it
is built with `accumulate="storage"`, which contracts a float32 table in
float32 at about twice the speed.

- `render_points(source_xy_lod, flux)` takes positions `[Q, 2]` and fluxes
  `[Q]`. Positions may move by `margin_nodes` node spacings (default 1) from
  those given at construction before the render refuses, which gives a fit
  room to move its sources. A source with zero flux is padding and is ignored.
- `render_raster(scene, x_lod, y_lod)` takes a scene image on a separable
  raster of source positions.
- Both return an `ImageRead` in delivered units per unit flux. The whole image
  is refused if any contributing source cannot be read. The error bound is
  unknown, as for a single read.

## What did this page establish?

- A read from 49 coefficient images reproduced direct propagation of the toy
  to {glue}`p_worst` of the local peak at worst over 40 positions.
- The tap count is chosen at load, by `plan_prefilter` from a requested tone
  error or directly, and the receipt records the design that was built.
- The grid must be sampled at $\rho\le0.64$, and reads within 6 nodes plus
  half a stencil of a table end are refused.
- The evidence a read carries is unchanged: sampling admitted, bound unknown.
- A scene of many sources renders as one contraction over the nodes it
  touches, equal to the sum of its reads.

What a prefiltered read costs in time is on {doc}`../reference/performance`,
and the exact rules are in {doc}`../reference/numerical-contract`.

## References

- A. Dutt and V. Rokhlin, "Fast Fourier transforms for nonequispaced data,"
  SIAM Journal on Scientific Computing 14(6), 1368-1393 (1993).
- J. A. Fessler and B. P. Sutton, "Nonuniform fast Fourier transforms using
  min-max interpolation," IEEE Transactions on Signal Processing 51(2),
  560-574 (2003).
- A. H. Barnett, J. Magland and L. af Klinteberg, "A parallel nonuniform fast
  Fourier transform library based on an exponential of semicircle kernel,"
  SIAM Journal on Scientific Computing 41(5), C479-C504 (2019).
