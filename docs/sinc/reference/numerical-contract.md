---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  render_markdown_format: myst
---

# Numerical contract

This page states exactly what a version 0.1 read computes and which inputs
admission accepts. The theory pages explain why, and this page is the
specification a test or a downstream library can rely on.

## Kernel and stencil

The kernel profile is `kaiser_w16_beta12_raw_v1`. It is the cut and tapered
sinc of {doc}`../foundations/sinc-interpolation`, where a sample $n$ receives
the weight $w(t-n)$ for a query $t$. Writing $u=t-n$ for the signed distance
in node spacings from the sample to the query, the weight of that tap is

$$
w(u)=\operatorname{sinc}(u)\,\frac{I_0\!\left(12\sqrt{1-(u/16)^2}\right)}{I_0(12)}
\quad\text{for } |u|<16,\qquad w(u)=0 \text{ otherwise},
$$

where $\operatorname{sinc}(u)=\sin(\pi u)/(\pi u)$ and $I_0$ is the modified
Bessel function of order zero. The weights are raw: they are not rescaled to
sum to one, and no flux, unit or photon-count normalization is applied.

For a query $s$ on an axis with lattice origin $o$ and node spacing $h$, the
query in node spacings is $t=(s-o)/h$, the same $t$ as on the foundations
pages. The stencil is the 32 taps with signed node indices `floor(t) - 15`
through `floor(t) + 16`: 16 at or below the query and 16 above. The two axes
combine as a separable product, a 32 by 32 block of 1024 taps per pixel
({doc}`glossary`). Every tap, including the zero-weight endpoints, must be
present. There is no exact-node shortcut, wrapping, zero filling, shortened
stencil or detector shift. Reads keep signed values, including small negative
excursions.

The cell below builds one stencil with the library and checks it against the
formula above for a seeded random query.

```{code-cell} python
:tags: [remove-cell]
import jax
import jax.numpy as jnp
import numpy as np
from myst_nb import glue

from yippy.sinc.kernel import stencil

jax.config.update("jax_enable_x64", True)
```

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(7)
spacing, origin = 0.38, 0.0
query = float(rng.uniform(6.0, 9.0))
taps, weights = stencil(jnp.asarray([query]), origin, spacing)
taps, weights = np.asarray(taps[0]), np.asarray(weights[0])
t = (query - origin) / spacing
u = t - taps
by_formula = np.sinc(u) * np.i0(12 * np.sqrt(1 - np.minimum(np.abs(u) / 16, 1) ** 2)) / np.i0(12)
by_formula[np.abs(u) >= 16] = 0.0
glue("c_query", f"{query:.3f}", display=False)
glue("c_t", f"{t:.3f}", display=False)
glue("c_first", f"{taps[0]}", display=False)
glue("c_last", f"{taps[-1]}", display=False)
glue("c_count", f"{taps.size}", display=False)
glue("c_diff", f"{np.max(np.abs(weights - by_formula)):.1e}", display=False)
glue("c_sum", f"{weights.sum():.6f}", display=False)
glue("c_max", f"{weights.max():.3f}", display=False)
glue("c_negative", f"{(weights < 0).sum()}", display=False)
```

For the query $s=${glue:text}`c_query` at spacing $h=0.38$, $t=${glue:text}`c_t`,
so the stencil holds {glue:text}`c_count` taps from node {glue:text}`c_first` to
node {glue:text}`c_last`. The library weights match the formula to
{glue:text}`c_diff`, they sum to {glue:text}`c_sum`, so a constant signal is reproduced to about
that accuracy, the largest weight is
{glue:text}`c_max`, and {glue:text}`c_negative` of them are negative.

## Floor rule

The floor rule is `exact_quotient_floor_v1`. For a host query, binary64
subnormals flush to signed zero, the difference $d=s-o$ is formed once in
binary64, and the C `fmod`
remainder $r=\operatorname{fmod}(d,h)$, which takes the sign of $d$ and is
exact in binary floating point, determines the floor of the exact real quotient: $n=\operatorname{round}((d-r)/h)$,
minus one when $r\ne0$ and $d<0$. JAX queries select taps on a common
binary32 coordinate grid in both lanes. The weights use the local phase at
the reader's working precision. See {doc}`stencil-floor` for the edge cases.

## Supported numeric range

Each positive lattice spacing and each nonzero origin component must have
magnitude in $[10^{-6},10^{6}]$ (a zero origin is allowed), box limits must
have magnitude at most $10^6$, and annulus radii must lie in
$[10^{-6},10^{6}]$, with zero allowed only for the inner radius. Admission
refuses out-of-range lattice metadata with `LAYOUT_UNSUPPORTED` and
out-of-range domain metadata with `METADATA_CONFLICT`.

Inside this range the floor rule is exact. After the subnormal flush, a
nonzero difference $s-o$ and a nonzero remainder are normal numbers:
cancellation near a nonzero origin and remainders at or above the minimum
spacing have binary resolution of at least $2^{-73}$ (a spacing of at least
$10^{-6}\approx2^{-20}$ and a 53-bit mantissa), and below the spacing
the remainder is the difference itself. Permitted quotients have magnitude at
most $2\times10^{12}$ (a difference of at most $2\times10^{6}$ divided by a
spacing of at least $10^{-6}$), below $2^{53}$, so recovering the integer and
correcting negative values are exact. Squares of supported radii and
boundary-scale coordinates neither underflow nor overflow. Queries far
outside the lattice fail coverage before any gathering.

## Permission and coverage

A read is permitted when the signed query at its decision resolution lies in the declared
`Domain` (a box or an origin-centered annulus, closed or lower-closed and
upper-open, never expanded by a tolerance) and every stencil node exists. A
safe point for compiled fallbacks is chosen from coverage endpoints, box
corners and centers, and annulus corner rays, side intersections and axis
crossings, using the same membership and floor rules as reads. An exotic
intersection that the finite candidate list cannot resolve refuses with
`INCOMPLETE_STENCIL` rather than expanding the domain.

## Quadrant grids

A quadrant grid has origin $(0,0)$, at least 17 nodes per axis (a read at the
origin needs nodes 0 through 16, and negative nodes are read from their
mirrored positives; a full signed grid needs a complete 32-node stencil, so at
least 32 per axis), an axis-aligned detector of `nx` by `ny` pixels centered
at `((nx - 1)/2, (ny - 1)/2)`, the reflection
kind `two_axis_pixel_flip_v1` and a `MirrorCheckSet` with x-negative,
y-negative and both-negative counterparts for each product. Negative taps
read the stored image at the absolute index, with detector columns reversed
for a negative x index and rows reversed for a negative y index.

Admission computes every mirror residual and every stored-axis invariance
check itself, and caller-supplied `check_passed` flags carry no authority.
Their local peak is recomputed from the images. The interior checks differ:
their residual is divided by the `core_peak` the delivery supplies, which
admission does not recompute, and the residual is recorded without gating
admission.

The versioned relative tolerance is $10^{-6}$ of the local peak (the largest
direct pixel value among pixel centers within radius 1 $\lambda_0/D$ of the
source). `Reflection.tolerance` may be stricter, $0<\text{tolerance}\le10^{-6}$.
A looser tolerance refuses with `SYMMETRY_UNSUPPORTED` before any comparison.
An axis image passes if it is exactly invariant, or if its relative residual
against a positive reference is within tolerance, and a nonzero discrepancy
without a positive reference refuses. The reference of an axis image is its
local peak, or its own maximum when its source lies more than 1 $\lambda_0/D$
from every pixel center (a halo node beyond the detector); that maximum is
below the off-frame peak, so the test is stricter. The receipt records
`axis_exact` separately and stores NaN in `axis_relative` where no positive
reference exists. Full grids require reflection kind `none` and no mirror
checks.

## Sampling admission

For every product and axis, with reference wavelength $\lambda_0$, declared
minimum sampling wavelength $\lambda_{\min}$ (and the optional campaign
minimum), declared support width $L$ in units of $D$ and spacing $h$ in
$\lambda_0/D$, admission computes the oversampling factor
$\rho=2(\lambda_0/\lambda_{\min})Lh$ ({doc}`../foundations/sampling`) in
binary64. $\rho\ge1$ adds `NYQUIST_FAIL` and $\rho>0.76$ (the limit of the
{doc}`glossary`) adds `KERNEL_MARGIN_FAIL`. Neither test implies accuracy at any position.

## Prefiltered profile

`PrefilteredKernelProfile(taps=7)` is a second profile for the same grids,
named `kaiser_bessel_prefiltered_v1`. Its identity string carries the tap
count, the filter half-width and the halo, for example
`kaiser_bessel_prefiltered_v1:taps7:filter12:halo6:exact_quotient_floor_v1`.
{doc}`../practice/prefiltered-read` introduces it. The rules that differ from
the raw profile are these.

- **Taps.** 5 to 10 per axis, chosen when the grid is admitted. Admission
  designs the filter, and the reader builds the coefficient images when it is
  constructed. The delivery
  holds raw samples for every choice.
- **Stencil.** With `floor` the exact quotient floor of the raw profile, an
  even tap count uses signed nodes `floor - (taps/2 - 1)` through
  `floor + taps/2`. An odd count centers on the nearest node, with an exact
  half rounding up: `center - (taps-1)/2` through `center + (taps-1)/2`. The
  half-cell test uses the same exact remainder as the floor.
- **Weights.** A tap at signed distance $u$ has weight
  $I_0\!\left(\beta\sqrt{1-(2u/\mathrm{taps})^2}\right)/\left(I_0(\beta)\,c\right)$
  for $|u|<\mathrm{taps}/2$ and zero otherwise, with the shape $\beta$ and the
  normalization $c$ of the design. The weight at the cut is $1/(I_0(\beta)\,c)$,
  not zero, so a read is discontinuous by about that fraction where the
  stencil moves by one node.
- **Coefficients.** The read applies the weights to coefficient images, the
  samples filtered along both source axes with the minimum-norm least-squares
  filter for the design band. The filter half-width is 12 nodes for 7 taps
  and 16 otherwise, the filter is one-sided near a table end, and the design
  runs in binary64. Coefficients are stored in the dtype of the samples and
  the read contracts in the reader precision. A read does not reproduce a stored sample
  exactly.
- **Design band.** The band on each axis is the largest $\rho/2$ over the
  products, in cycles per node, so the design holds at the shortest declared
  wavelength.
- **Coverage.** Every tap must lie at least 6 stored nodes (the halo) inside
  each table end. For a quadrant the mirrored nodes count across the axes.
  Coverage is decided from the integer first tap, by the same floor
  arithmetic as the read.
- **Sampling.** $\rho>0.64$ adds `KERNEL_MARGIN_FAIL`.
- **Minimum size.** 25 nodes per axis for 7 taps and 33 otherwise, or 13 and
  17 for a quadrant.
- **Receipt.** `Receipt.kernel_design` holds ordered `(name, value)` pairs:
  `taps`, `filter_half_width`, `halo`, and for each axis `band`, `beta`,
  `tone_error`, `interior_tone_error` and `sample_gain`, plus
  `tone_error_two_axis`, the sum of the two axis errors and their product.
  Tone errors are maxima over a sampled set of in-band frequencies and query
  phases. They are not bounds, and they exclude the rounding of coefficients
  to the storage dtype. The field is empty for the raw profile, whose receipt
  identities are unchanged.
- **Evidence.** Unchanged: accepted reads are `SAMPLING_ADMITTED` with an
  unknown bound, and a bounded policy refuses. Independent checks at
  off-node positions are evaluated with prefiltered reads. Mirror checks
  compare stored samples, as for the raw profile, and do not use the profile.
- **Readers.** `PrefilteredReader` requires a grid admitted with the same
  profile, and `Interpolator` requires one admitted with the raw profile.
  Both require a grid that stores samples. A `PrefilteredReader` keeps its
  coefficients in place of the samples: its `grid.intensity` is the
  coefficient table and its `grid.stored` is `"prefiltered_coefficients"`.
- **Planning.** `plan_prefilter(*, declared_lattice, products, reflection,
  domain, max_tone_error)` returns the fewest taps whose
  `tone_error_two_axis` is at or below the request and refuses with
  `TOLERANCE_EXCEEDED` when 10 taps are not enough.
- **Scenes.** `SceneRenderer(reader, product_index, source_xy_lod,
  margin_nodes=1, accumulate="float64")` renders
  `render_points(source_xy_lod, flux)` and `render_raster(scene, x_lod,
  y_lod)` as one contraction over a node set fixed at construction. On a
  mirror-symmetric grid the table holds one image per stored node, a quarter
  of the touched signed nodes; amplitudes fold per sign quadrant and the four
  partial images are flipped and added. With `accumulate="float64"` node
  amplitudes and the contraction are binary64 for float32 and float64
  storage alike (a float32 table is widened 1,024 detector columns at a
  time), so the result equals the flux-weighted sum of the single reads to
  binary64 rounding. With `accumulate="storage"` amplitudes are rounded to
  the storage dtype and the contraction accumulates in it. The whole image is
  refused when any source with nonzero flux cannot be read or lies outside
  the node set.
- **Memory.** The resident estimate is one table, the coefficients, which
  replace the samples. Scratch counts one `taps` by `taps` block of images and
  the load-time filter workspace, and `Tiling.pixel_tile` is not used. A scene
  renderer adds its table of stored touched images; a binary64 render over
  float32 coefficients counts one widened block of detector columns.

## Precision

Admission uses the original host coordinates in binary64. For Python and
NumPy queries the reader checks permission, selects taps and subtracts the
lattice origin before narrowing. Pass translated coordinates in this form
when their separation is smaller than one binary32 spacing at the origin.
JAX arrays, including arguments to `jit`, use a common binary32 grid for
permission and tap selection in both lanes. Weights and contractions use
the precision selected at reader construction.

The prefilter operates on the delivered samples in host binary64.
Each coefficient chunk is checked before rounding once into storage.
An overflow or an underflow to zero raises `ValueError` during
construction. An explicit
`accumulate="float64"` scene request requires a float64 reader and raises
`hwoutils.PrecisionError` when unavailable. The default uses float64 with a
float64 reader and storage precision with a float32 reader.

Adapters that converted float32 coordinates to float64 angular units pass
`offset_original_dtype=jnp.float32`. The four-ULP coordinate allowance then
uses `nextafter` in that original dtype at the converted scale.

## Compilation, identity and provenance

`read_batch(reader, source_xy_lod, product_index)` takes `[Q, 2]`
positions and int32 `[Q]` product indices and maps serial chunks. Static
metadata stays unbatched, and an empty batch performs no interpolation.
`eqx.filter_jit` and an outer `jax.vmap` are supported, but an outer mapped
dimension needs the caller's own memory accounting.

A readmitted product has
a content-bound static `receipt_id` and compiles once for that identity, while
replacing array leaves of the same shape without readmission does not
recompile. Readmit modified samples before reuse so that provenance and
empirical evidence stay valid. `provenance=(("key", "text"), ...)` pairs are
kept verbatim in the receipt and hashed into its identity, and they are preserved,
not validated.

## Memory accounting

`MemoryBudget` has four required ceilings in bytes, with no defaults:
`max_resident_bytes`, `max_output_bytes`, `max_scratch_bytes` and
`max_total_bytes`. They cover native resident samples, output
leaves and estimated tiled scratch, and the total also counts the two extra
native copies that admission stages for reordering. Admission validates
completeness and duplicates with numerical linear indices and occupancy
counts, with no source-sized Python containers, and its estimate includes
coordinate and index casts, residual arrays, linear indices, occupancy,
sorting and the canonical reorder. These are allocation estimates, not
measured compiler workspace or whole-process peaks. The library does no file
I/O, spectral interpolation, fallback interpolation or certificate
evaluation.
