# Spatial circle-reader accuracy

`azimuthal_mean_profile` in `src/yippy/performance.py` defaults to the mean of a
**Keys cubic interpolant** on each requested circle. Pass
`interpolation="quintic"` to opt into a degree-5 cardinal B-spline. It does not promise the
mean of the continuous optical intensity that was sampled to make the image.
Increasing the angle count converges the angular integral of that interpolant;
it does not remove spatial interpolation error. The default Keys calculations and persistent performance-cache identities
remain unchanged.

**Verdict: this is an accuracy limit of the chosen spatial interpolant, not a
center, axis-order, unit, or indexing bug.** Use the opt-in quintic reader when sub-percent circle means are needed on
the coarse grid, with an absolute-error criterion near nulls and validation
against the original optical model. Changing the default remains an owner
decision because downstream cached products depend on that choice.

## Select the spatial reader

```python notest
import equinox as eqx
from yippy.performance import azimuthal_mean_profile

read = eqx.filter_jit(azimuthal_mean_profile)
profile = read(image, center, radii_pix, n_theta=256, interpolation="quintic")
```

`interpolation` is a static string under JIT; with raw `jax.jit`, include it
and `n_theta` in `static_argnames`. Enable JAX float64 for the precision
reported here. The option is also available on `core_mean_intensity_profile`,
`compute_core_mean_intensity_curve`, `Coronagraph.core_mean_intensity_curve`,
and both `Coronagraph.core_mean_intensity` and
`EqxCoronagraph.core_mean_intensity`. The intensity map and AYO noise-floor
helpers forward it. This controls spatial interpolation; `interp_order`
still controls the subsequent interpolation between radial profile knots.

The quintic implementation is pure JAX, with no SciPy at read time. It follows
P. Thevenaz, T. Blu and M. Unser, "Interpolation Revisited", *IEEE Transactions
on Medical Imaging* **19**(7), 739-758 (2000), doi:10.1109/42.875199.
Two causal/anticausal recursive filters per axis solve for the cardinal
B-spline coefficients once per image per call. All radii and tangent endpoints
reuse those coefficients, with a separable six-by-six evaluation stencil.
The prefilter itself is JIT-compiled for reuse across calls of the same shape
and dtype; image-dependent coefficients are recomputed when the image changes.

Nearest-edge extension applies to the **input samples**, before prefiltering.
A 48-pixel halo suppresses finite filter initialization effects below float64
roundoff at the original image (the largest pole has magnitude 0.431).
Stencil reads retain this halo. Clamping the coefficients themselves would
be a different boundary condition. Angular normalization is unchanged:
only in-image arcs contribute, tangencies use endpoint values, and circles
entirely outside return zero. No high-order accuracy is promised at an edge,
or for negative lobes introduced by interpolation near a null.

`Coronagraph` keeps on-demand profiles separately by reader and diameter.
Quintic calls recompute from the stellar map when a precomputed Keys profile
would otherwise be used, including point sources and the single-profile fast path.
`compute_all_performance_curves(interpolation="quintic")` can explicitly
rebuild the in-memory core intensity tables; later calls still default to
Keys and select or compute the appropriate profile. These core intensity
tables are rebuilt from maps, not read from or written to the performance
FITS cache. Throughput, raw contrast and persistent cache identities are
unchanged. Consumers saving their own quintic-derived products must distinguish
them from Keys-derived products.

## Reproduce without data files

From the repository root, using an environment with yippy's dependencies:

```sh
JAX_PLATFORMS=cpu PYTHONPATH=src python tools/check_circle_reader.py
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_circle_reader.py
```

The diagnostic prints the radius scan as CSV to stdout. Coordinate checks,
alternative-reader results, and timings go to stderr. It builds two analytic
maps in float64:

- A radial Airy intensity, `I(r) = [2 J1(pi r)/(pi r)]**2`, whose full-circle
  mean is exactly `I(r)` (the value at the origin is its limit, one).
- A displaced Gaussian, `I(x,y) = exp(-((x-1)**2 + (y+2)**2)/18)`, whose
  mean is `exp(-(r**2+5)/18) I0(r sqrt(5)/9)`. This map is not symmetric
  about the circle center.

The first grid is 128 by 128, centered at pixel index `(63.5, 63.5)`, with
0.25 lambda/D per pixel. Values are point samples multiplied by pixel area,
not integrals over pixel footprints. The reader's result is divided by that
same area once. A second grid halves the pixel scale while keeping the physical
field of view fixed. The script compares the default angle count (256 on the
first grid, 512 on the second), 2048 angles, independently constructed
2048-angle coordinates evaluated with Keys, the JAX quintic reader at both
angle counts, and an independent SciPy quintic bivariate spline. The complete
measured scan is in [circle_reader_accuracy.csv](circle_reader_accuracy.csv).

## Coordinate and boundary checks

- `stellar_axis_index` converts corner-based metadata to pixel-center indices
  by subtracting 0.5: metadata center 64 becomes index 63.5.
- The reader accepts radii in pixels. The callers convert lambda/D to pixels
  by dividing by the pixel scale; the reader neither rescales intensity nor
  integrates pixel footprints.
- The regression tests use an unequal quadratic in x and y plus linear terms,
  at both the central and an off-axis circle center. Its analytic mean tests
  the center convention, row/column ordering and supplied unit conversions.
  It does not exercise the loader's entire unit-conversion path; that path
  was inspected separately.
- Radii 3, 5 and 10 lambda/D are 12, 20 and 40 pixels on the first grid. Their
  four-tap stencils are entirely inside the image. Boundary handling has no
  effect on these cases.
- For circles crossing the image boundary, the reader averages only in-image
  arc length; it does not zero-pad the missing arc. Stencil samples beyond an
  edge use nearest-edge extension. A tangent circle uses its in-image endpoint
  values, and a circle entirely outside returns zero. The constant-map test
  covers short corner arcs, tangency and the exterior. This does not establish
  high-order accuracy for nonconstant maps at an edge.
- The spatial reader does not consult the radial-profile knots. The profile
  spacing is 0.25 pixel, or 0.0625 lambda/D on the first grid. The diagnostic
  includes both knots and intervening radii. Exact knots avoid a subsequent
  radial interpolation error but still require spatial interpolation around
  the circle.

## Measured interpolation error

The analytic Airy map exposes percent-level spatial error even though its
circle mean is known exactly. Signed relative errors below are percentages
against the analytic mean, not against the quintic interpolant. The tables below preserve the original study
measurements (the quintic columns used SciPy). The new JAX quintic reader
reproduces those interior spline results; the CSV records both implementations
and both angle counts. At 3, 5 and 10 lambda/D its coarse-grid errors are
0.0844264%, 0.0672181% and 0.0387699%, and its fine-grid errors are
0.000600227%, 0.000424109% and 0.000228349%. Tests require each absolute
percentage error to be less than twice the corresponding study measurement,
for both Airy and displaced Gaussian maps.

The coarse-grid study results at 0.25 lambda/D per pixel are:

| Radius (lambda/D) | Keys, 256 angles | Keys, 2048 angles | Quintic, 2048 angles |
| ---: | ---: | ---: | ---: |
| 2.5 | -3.91979 | -3.92109 | -0.0689284 |
| 3 | 3.98183 | 3.97591 | 0.0844264 |
| 3.03125 | 5.49883 | 5.50263 | 0.118276 |
| 3.23 | 3054.57 | 3054.58 | 22.8071 |
| 3.25 | 1580.01 | 1581.03 | 11.5656 |
| 4 | 3.07187 | 3.07827 | 0.0767517 |
| 5 | 2.52420 | 2.51103 | 0.0672181 |
| 5.03125 | 3.76466 | 3.76867 | 0.0953597 |
| 6 | 2.12272 | 2.12455 | 0.0589771 |
| 7 | 1.84132 | 1.85674 | 0.0522745 |
| 8 | 1.57987 | 1.64801 | 0.0468531 |
| 9 | 1.52910 | 1.48170 | 0.0424282 |
| 10 | 1.34473 | 1.35708 | 0.0387699 |
| 10.03125 | 2.51703 | 2.45205 | 0.0609021 |
| 11 | 1.15308 | 1.24876 | 0.0357048 |

The inter-knot radii 3.03125, 5.03125 and 10.03125 have larger errors than
their adjacent integer radii. The error is not confined to profile knots.
Relative error becomes especially large near Airy zeros; a higher-order
interpolant also needs an error budget there.

Halving the pixel scale to 0.125 lambda/D per pixel gives the following Airy
errors. This uses new samples of the analytic optical intensity, not a resize
of the coarse image:

| Radius (lambda/D) | Keys, 512 angles | Keys, 2048 angles | Quintic, 2048 angles |
| ---: | ---: | ---: | ---: |
| 2.5 | -0.284681 | -0.284425 | -0.000527569 |
| 3 | 0.292769 | 0.293174 | 0.000600227 |
| 3.03125 | 0.300829 | 0.300892 | 0.000805548 |
| 3.23 | 189.602 | 189.552 | 0.255601 |
| 3.25 | 103.808 | 103.909 | 0.137583 |
| 4 | 0.231568 | 0.230392 | 0.000504066 |
| 5 | 0.187675 | 0.190011 | 0.000424109 |
| 5.03125 | 0.199210 | 0.198263 | 0.000588839 |
| 6 | 0.157854 | 0.162196 | 0.000363288 |
| 7 | 0.140899 | 0.142020 | 0.000316838 |
| 8 | 0.132704 | 0.126452 | 0.000280636 |
| 9 | 0.119994 | 0.114370 | 0.000251794 |
| 10 | 0.100663 | 0.104803 | 0.000228349 |
| 10.03125 | 0.126008 | 0.126330 | 0.000368319 |
| 11 | 0.0901818 | 0.0968365 | 0.000208952 |

The broad displaced Gaussian is much less sensitive. Signed percentage errors
at 2048 angles are:

| Radius (lambda/D) | Keys, scale 0.25 | Quintic, scale 0.25 | Keys, scale 0.125 | Quintic, scale 0.125 |
| ---: | ---: | ---: | ---: | ---: |
| 2.5 | 0.0000231635 | -5.43340e-9 | 0.00000494392 | -8.50209e-11 |
| 3 | 0.0000624349 | -2.91480e-9 | 0.00000742804 | -4.75842e-11 |
| 3.03125 | -0.0000631584 | -2.78452e-9 | -0.0000164584 | -4.65850e-11 |
| 3.23 | 0.000127926 | -1.71995e-9 | 0.0000136445 | -3.22520e-11 |
| 3.25 | 0.0000824929 | -1.41681e-9 | 0.00000865377 | -2.51799e-11 |
| 4 | 0.000142157 | 3.78766e-9 | 0.0000120277 | 5.32241e-11 |
| 5 | 0.000208256 | 1.07434e-8 | 0.0000150207 | 1.60183e-10 |
| 5.03125 | 0.000134873 | 1.15226e-8 | -0.000000214125 | 1.63980e-10 |
| 6 | 0.000228873 | 1.37835e-8 | 0.0000142824 | 2.10987e-10 |
| 7 | 0.000156495 | 8.01301e-9 | 0.00000660013 | 1.30607e-10 |
| 8 | -0.0000730696 | -9.62731e-9 | -0.0000121737 | -1.31695e-10 |
| 9 | -0.000537429 | -3.69529e-8 | -0.0000469357 | -5.49960e-10 |
| 10 | -0.00132996 | -6.25964e-8 | -0.000103320 | -9.59399e-10 |
| 10.03125 | -0.000612115 | -6.67755e-8 | 0.0000394664 | -9.84146e-10 |
| 11 | -0.00255493 | -6.17551e-8 | -0.000188372 | -9.86522e-10 |

At 3, 5 and 10 lambda/D the default Gaussian reader gives 0.00006564%,
0.00022264% and -0.00144605% on the coarse grid, and 0.000007147%,
0.00001431% and -0.0001016% on the fine grid. These different errors on the
same grids support a spatial interpolation limit, rather than a universal
center or normalization offset. The smooth-map regression requires agreement
with the closed-form mean to relative tolerance 5e-5 (0.005%) at radii 0, 3,
3.03125, 5 and 10 on the coarse grid. It does not pin the measured residuals
or promise that tolerance for oscillatory optical maps.

For every map and scale, the production reader at 2048 angles agrees with the
independent Keys coordinates to at most 5.6e-16 in relative value. Increasing
the number of angles changes only angular quadrature of the same interpolant.

### Why percent-level error is plausible

At radius 3 the Airy mean is 0.00140642 of peak and the coarse Keys absolute
error is about 0.0000559 of peak, giving 3.98% relative error. At radius
3.03125 the mean drops to 0.00107876 and the absolute error is about 0.0000594,
giving 5.50%. At 3.23, close to a null, the mean is only 0.00000165692; an
absolute error of about 0.0000506 gives 3055% relative error. Even quintic
has a 22.8% relative error there, despite much smaller absolute error.

Thus 2 to 8 percent is entirely plausible at 0.25 lambda/D per pixel where
the intensity changes rapidly on a scale comparable to the pixel spacing and
is small relative to nearby intensities. The interpolation residual is divided
by a small circle mean. Small intensity alone is insufficient: multiplying a
map by a constant leaves its relative error unchanged. The important condition
is large local variation or curvature relative to that mean. These synthetic
maps establish the mechanism and its magnitude; they do not reproduce the
specific reported 7.8%, 0.55% and 2.2% measurements at 3, 5 and 10 lambda/D
on the optical design, whose map and direct propagation were not used here.

## Options and cached products

Keys remains the default. The diagnostic compares the opt-in quintic reader
with the following alternative sampling strategies:

1. **Quintic spline on the existing grid (available opt-in).** At radii
   3, 5 and 10 its errors are 0.08443%, 0.06722% and 0.03877%, versus Keys'
   3.97591%, 2.51103% and 1.35708%. This is the most accurate measured
   point-circle read on the coarse samples. The JAX implementation supplies coefficient construction and
   nearest-edge sample extension. It still needs validation near nulls; sub-percent error is not guaranteed
   arbitrarily close to a null.
2. **Finer samples before the read.** Regenerating the map at scale 0.125
   reduces Keys' errors at those radii to 0.29317%, 0.19001% and 0.10480%.
   The 256 by 256 map has four times as many pixels as the 128 by 128 map
   (512 versus 128 KiB in float64), and the default angle count doubles.
   Actual optical propagation cost is not measured by these analytic maps.
   Resizing an existing coarse map does not create the missing optical
   information. The script also measures quintic upsampling of that coarse
   image followed by a fine-grid Keys read; its residual includes both the
   coarse spline error and the final Keys error: 0.36930%, 0.25243% and
   0.14169% at radii 3, 5 and 10. Reading the coarse quintic directly is more
   accurate and avoids the additional fine image. Upsampling with the same
   Keys kernel cannot be assumed to restore accuracy either.
3. **Integrate the pixel footprint if detector-area means are the intended
   quantity.** The prototype averages the coarse quintic interpolant over an
   axis-aligned square of side 0.25 lambda/D at each circle location, using
   8 by 8 Gauss-Legendre quadrature, then averages over angles. It divides by
   pixel area, so this is a mean density. Its reference integrates the analytic
   Airy intensity over the same footprints. This changes the estimand from
   intensity on an infinitesimal circle to a square-smoothed circle mean;
   it cannot repair an error against a point-circle reference. It is also
   distinct from an annular average. Input pixels that already store footprint
   integrals must not be integrated a second time without accounting for that
   response.

The footprint comparison makes the difference in target explicit:

| Radius (lambda/D) | Analytic footprint mean versus point-circle mean (%) | Quintic footprint error versus analytic footprint mean (%) |
| ---: | ---: | ---: |
| 3 | 3.86247 | 0.0608258 |
| 5 | 2.18673 | 0.0475280 |
| 10 | 1.04520 | 0.0268545 |
| 3.23 (near null) | 7383.99 | 0.361267 |

The 8 by 8 versus 12 by 12 analytic footprint integrals agree to 2.5e-15
relative over the entire scan. Footprint averaging reduces sensitivity to a
null by including adjacent intensities, but the resulting mean is different
from the original point-circle reference even with exact optical samples.

### JAX quintic accuracy and cost

The implemented readers give these signed percentage errors at 2048 angles.
The CSV contains all 15 radii, both maps and scales, default-angle results,
and the independent SciPy spline comparison.

| Map | Pixel scale (lambda/D) | Radius (lambda/D) | Keys error (%) | JAX quintic error (%) |
| --- | ---: | ---: | ---: | ---: |
| Airy | 0.25 | 3 | 3.97591 | 0.0844264 |
| Airy | 0.25 | 5 | 2.51103 | 0.0672181 |
| Airy | 0.25 | 10 | 1.35708 | 0.0387699 |
| Airy | 0.125 | 3 | 0.293174 | 0.000600227 |
| Airy | 0.125 | 5 | 0.190011 | 0.000424109 |
| Airy | 0.125 | 10 | 0.104803 | 0.000228349 |
| Gaussian | 0.25 | 3 | 0.0000624349 | -2.91480e-9 |
| Gaussian | 0.25 | 5 | 0.000208256 | 1.07434e-8 |
| Gaussian | 0.25 | 10 | -0.00132996 | -6.25956e-8 |
| Gaussian | 0.125 | 3 | 0.00000742804 | -4.75842e-11 |
| Gaussian | 0.125 | 5 | 0.0000150207 | 1.60183e-10 |
| Gaussian | 0.125 | 10 | -0.000103320 | -9.59421e-10 |

At the default 256 angles, the coarse-grid quintic Airy errors at 3, 5 and
10 are 0.0844265%, 0.0672190% and 0.0387492%. Thus the accuracy improvement
also holds without raising the angle count. Near the 3.23 lambda/D null,
the coarse-grid quintic error is still 22.8%; use an absolute error budget
there and validate against optical propagation.

A CPU run on 2026-11-04 used arm64, Python 3.12.12, JAX 0.12.0.1, float64
and four threads. Each row evaluates the same 15 radii on a 128 by 128 map
with 256 angles. Warm times are medians of seven synchronized calls.
Compilation is measured with `lower(...).compile()`; tracing/lowering is
reported separately. The first execution is also synchronized.

| Reader | Trace/lower (ms) | Compile (ms) | First execution (ms) | Warm JIT (ms) | Warm eager (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Keys | 22.28 | 104.85 | 0.52 | 0.1768 | 1.1818 |
| Quintic | 37.47 | 136.34 | 2.94 | 0.8707 | 4.3667 |

These end-to-end quintic times include computing coefficients for the image,
arc construction, spatial evaluation, and reduction for every call. The
quintic prefilter has a reusable JIT executable but does not cache image
values. Its padded coefficient array has 224 by 224 entries (392 KiB in
float64), compared with the 128 KiB input. Quintic's warm JIT cost was about
4.9 times Keys in this run. Timings on this shared host are indicative,
not portable speed ratios; no GPU or optical-propagation cost was measured.

The reader regressions cover both analytic maps at both scales, exact
pre-change CPU Keys bits, nearest extension for nonconstant edge samples,
short arcs and tangencies, and small/degenerate images. Reverse-mode image
gradients are finite; center/radius gradients away from tangencies also
pass a finite-difference check. Arc topology changes at tangency are not
smooth. All 17 reader tests passed in 4.52 s, with the slowest test at 1.25 s.
Public-caller tests cover both radial interpolation modes, cache separation,
point sources, extended sources, the single-profile fast path and rebuilding
the in-memory tables at a nonzero default diameter. All 12 caller tests
passed in 8.98 s; the slowest test call was 2.60 s. The single-profile fixture
uses valid spline support knots because the existing stellar FITS loader
requires at least two diameter knots.

### Original study cost

One local run used an arm64 CPU, Python 3.12.12, JAX 0.12.0.1, float64, and
a four-thread limit. The following milliseconds are medians of seven warmed
calls for the same 15 radii. JAX results are synchronized before stopping the
timer. This is a shared host: timings, especially compilation and JIT dispatch,
are indicative measurements, not portable speed ratios. No GPU cost or optical
propagation cost was measured.

| Reader and grid | Angles | Eager read (ms) | Warm JIT read (ms) | JIT compile plus first read (ms) |
| --- | ---: | ---: | ---: | ---: |
| Keys, 128 by 128 | 256 | 1.687 | 0.164 | 255 |
| Keys, 128 by 128 | 2048 | 2.118 | 9.701 | 395 |
| Keys, 256 by 256 | 512 | 4.210 | 0.720 | 294 |
| Keys, 256 by 256 | 2048 | 6.657 | 2.576 | 321 |

| Alternative | Preparation (ms) | Read (ms) | Storage/work beyond original map |
| --- | ---: | ---: | --- |
| SciPy quintic, 128 by 128, 2048 angles | 1.850 | 5.474 | Spline coefficients |
| SciPy quintic, 256 by 256, 2048 angles | 5.851 | 7.410 | Four times the input pixels, plus coefficients |
| Coarse quintic to 256 by 256, then Keys, 2048 angles | 1.850 + 3.382 | 6.657 eager / 2.576 JIT | Additional 512 KiB fine map, plus coefficients |
| Coarse quintic footprint integration, 2048 angles | 1.850 | 567.903 | 64 spline evaluations per angular sample |

Spline preparation is measured separately and can be reused across circle
reads. SciPy evaluation uses precomputed angular coordinates, whereas the Keys
calls include construction of in-image arcs and coordinates. The original study did not benchmark a JIT-compatible quintic reader;
the new implementation is measured separately below. The prototype spline footprint
read is about 104 times the coarse point-spline read in this run, with 64
quadrature samples per footprint. Analytic Airy map generation itself takes
0.334 ms on the coarse grid and 1.291 ms on the fine grid; these are sampling
costs of this analytic example, not estimates of a coronagraph propagation.

For an existing coarse map and a point-circle target, the measured accuracy
and preparation cost favor a direct quintic reader over quintic resizing plus
Keys. Regenerating finer optical samples remains valuable when feasible and
can be combined with quintic: the fine-grid quintic errors at 3, 5 and 10 are
0.000600%, 0.000424% and 0.000228%. Integrating footprints is appropriate when
the physical quantity calls for it, rather than as a substitute for spatial
interpolation accuracy.

More angles alone do not resolve a spatial interpolation limit. An absolute
error budget alongside relative error is useful near nulls, but does not
justify silently relaxing a requested relative-accuracy requirement.

Existing stellar maps, performance files, cache keys and default computed
outputs remain unchanged. Explicit quintic reads change core-intensity
profiles and quantities a consumer derives from them. Switching the default
would require a separate decision and review of downstream cached products.
