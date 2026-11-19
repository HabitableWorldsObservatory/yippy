# Glossary

Terms as yippy.sinc uses them, each with the page that develops it. The sampling
terms (sample, frequency, bandwidth, Nyquist limit, aliasing, weight, query,
tap, stencil) are defined on {doc}`../foundations/sampling` and
{doc}`../foundations/sinc-interpolation`, and this page follows those
definitions.

$\sigma_{\max}$
: The largest value of $\lambda_0/\lambda$ among the wavelengths a product
  contains, 1 for a monochromatic product at $\lambda_0$.

$D$ and $\lambda_0/D$
: $D$ is the telescope entrance diameter (the declared width of the
  illuminated pupil sets the support width $L$ in units of $D$). $\lambda_0/D$
  is the angle that unit corresponds to at the reference wavelength
  $\lambda_0$, and every position, spacing and radius on these pages is
  measured in it.

admission
: The one-time check of a grid before any read: geometry, declared physical
  premises, sampling limit, symmetry evidence and memory. It returns an
  `Admission` holding either a usable grid or the reasons for refusal.
  See {doc}`../practice/admission`.

aliasing
: A cosine faster than the Nyquist limit looks, at the samples, like a slower
  cosine: frequency $f$ looks like $1-f$ cycles per sample, and nothing in the
  samples reveals it. A band limit below the Nyquist limit rules such twins
  out. See {doc}`../foundations/sampling`.

APLC
: Apodized pupil Lyot coronagraph: an apodizing mask in the pupil, a hard
  occulting disk in the focal plane and a Lyot stop. The LUVOIR-A APLC is the
  public design used on {doc}`../validation`.

band-limited
: A signal with nothing above its bandwidth. In a plot of its spectrum, the
  spectrum is zero everywhere to the right of a vertical line. A coronagraph's
  planet image at one detector pixel, as a function of source position, is
  band-limited with bandwidth $B=\sigma_{\max}L$. See
  {doc}`../foundations/sampling` and {doc}`../theory/band-limits`.

bandwidth
: A frequency above which a signal is known to contain nothing. It is a bound,
  and the signal need not reach it. Two forms are used. $B$ is measured in
  cycles per $\lambda_0/D$ of source motion, and $f_{\max}=Bh$ is the same
  frequency in cycles per sample, where $h$ is the node spacing in
  $\lambda_0/D$. The foundations pages use $f_{\max}$, and the theory and
  practice pages use $B$. See {doc}`../foundations/sampling`.

blend interpolation
: The existing interpolation in yippy of tabulated PSFs, which combines the
  four nearest delivered images, each shifted to the query. On the
  physicaloptix starter delivery its worst error is tens of percent of the
  local peak, against about one part in a million for yippy.sinc. See
  {doc}`../validation`.

cell
: The interval between two neighboring nodes along one axis. A query lies in
  one cell, and the stencil is chosen from the cell's index $\lfloor t\rfloor$.
  See {doc}`stencil-floor`.

centrosymmetric
: Unchanged by inversion through the center, $f(-\mathbf{x})=f(\mathbf{x})$.
  A circular occulting disk is centrosymmetric.

conjugation
: Complex conjugation of the optical field. For a real pupil, a real
  centrosymmetric focal-plane mask and a real stop, the field at the reversed
  source position is the conjugate of the reversed field, which supplies a
  mirror symmetry of the intensity. See {doc}`../theory/quadrant-symmetry`.

coefficient image
: A delivered image stack filtered along both source axes when the grid is
  loaded, one for each node. The prefiltered profile reads coefficient images
  with a short bump instead of reading samples with the sinc kernel. See
  {doc}`../practice/prefiltered-read`.

core flux
: The summed pixels whose centers lie within 0.7 $\lambda_0/D$ of the true
  source position, the number a yield code uses. Validation reports its
  relative error. See {doc}`../validation`.

cycles per sample
: The unit of frequency on the foundations pages. Positions are measured in
  node spacings, so a cosine with 4 samples per cycle has frequency 0.25
  cycles per sample. A frequency and its samples per cycle are reciprocals.

domain
: The `Domain` declaration: the region of source positions where reads are
  permitted, a box or an origin-centered annulus.

evidence code
: The integer that says how a premise was established, declared beside the
  premise. Support width must be declared with code 1. See
  {doc}`../practice/declarations`.

fiducial test
: A check that the star sits where the declared detector frame puts it: an
  image propagated for a known source position must centroid at the pixel the
  frame predicts. See {ref}`recipe-frame`.

fold
: Mapping a stencil tap at a negative node index onto the stored node it mirrors
  in a quadrant grid. See {doc}`../theory/quadrant-symmetry`.

full grid
: A delivery whose nodes cover both signs of each source coordinate, so no
  mirror symmetry is needed. Compare quadrant grid.

independent check
: A PSF propagated directly at a position between grid nodes and delivered
  separately, so a read can be compared with it. Checks are evidence about
  the delivered model at those positions, not a bound elsewhere.

inner working angle
: The smallest planet separation, in $\lambda_0/D$, at which a coronagraph
  design is meant to work. The public LUVOIR-A APLC design value is quoted on
  {doc}`../validation`, and the starter's checks lie around it.

interior check
: Another name for an independent check placed inside the served region.

Kaiser window
: A bell-shaped taper built from the modified Bessel function $I_0$ with a
  shape parameter $\beta$. The kernel multiplies the sinc by it, so the window
  falls to $1/I_0(12)$, about $5\times10^{-5}$ of its peak, at 16 samples,
  where the sinc has already reached zero. See {doc}`../theory/kernel`.

Kaiser-sinc kernel
: The sinc cut at 16 samples on each side and multiplied by a Kaiser window
  with $\beta=12$, yippy.sinc's interpolation shape. See {doc}`../theory/kernel`.

kernel
: The weight function $w(t)$ that turns nearby samples into a value between
  them: the shape $w$ of {doc}`../foundations/sinc-interpolation`, with the
  sinc cut at 16 samples and tapered by a Kaiser window with shape parameter
  $\beta=12$. The weight of one sample is the kernel evaluated at that
  sample's distance from the query. See {doc}`../theory/kernel`.

lattice
: The regular grid of source positions where PSFs were propagated, given by
  an origin, a spacing per axis and a node count per axis.

limit
: The largest oversampling factor the kernel is admitted at, $\rho=0.76$.
  Past it the kernel's worst-tone error rises by orders of magnitude, and
  admission refuses with `KERNEL_MARGIN_FAIL`. See {doc}`../theory/kernel`.

local peak
: The largest directly propagated pixel value among pixel centers within
  1 $\lambda_0/D$ of the source. Errors are quoted relative to it, so a
  number such as $10^{-6}$ means one millionth of the planet's own brightest
  pixel. In the admission interior checks the denominator is the delivered
  `core_peak`, which admission does not recompute, while the mirror and axis
  checks recompute theirs from the images.

margin
: The room left under the limit: $0.76-\rho$. A grid with $\rho=0.719$ has a
  margin of 0.041. A negative margin means the grid is refused.

mean density
: A pixel value equal to the pixel integral divided by the pixel area. See
  {ref}`recipe-pixel`.

mirror check
: An independent propagation at a negative source position, compared with
  the flipped stored image at the mirrored positive node. Admission requires
  them for quadrant grids. See {doc}`../theory/quadrant-symmetry`.

node
: One lattice position with its delivered PSF. A node is the 2-D version of
  what the foundations pages call a sample.

Nyquist limit
: A frequency, 0.5 cycles per sample, at which a cosine gets exactly two
  samples per cycle. Cosines faster than it alias. A band-limited signal is
  determined by its samples only if its bandwidth is strictly below the limit
  ($\rho<1$). See {doc}`../foundations/sampling`.

Nyquist spacing
: The node spacing at which the bandwidth sits exactly at the Nyquist limit,
  $1/(2B)$ in $\lambda_0/D$. A grid with oversampling factor $\rho$ has
  spacing $\rho$ times the Nyquist spacing.

oversampling factor
: $\rho=2f_{\max}=2Bh$, the bandwidth as a fraction of the Nyquist limit. A
  smaller $\rho$ means more samples per cycle than needed, so $\rho=0.5$
  means four samples per cycle at the bandwidth and $\rho=1$ is the Nyquist
  limit itself. yippy.sinc admits grids with $\rho\le0.76$ on every axis.
  See {doc}`../foundations/sampling`.

phase ramp
: The linear phase a tilted source puts across the pupil; its slope is set by
  the source position. See {doc}`../theory/band-limits`.

pixel integral
: A pixel value equal to the light falling on the whole pixel, the format a
  real delivery should use. See {ref}`recipe-pixel`.

point sample
: A pixel value equal to the intensity at the pixel center. The toy images on
  these pages are point samples. See {ref}`recipe-pixel`.

prefiltered read
: A read that sums 5 to 10 coefficient images per axis with Kaiser-Bessel
  bump weights. It is faster and less accurate than the raw 32-tap read and
  needs $\rho\le0.64$. See {doc}`../practice/prefiltered-read`.

premise
: One of six physical statements the band-limit argument needs (for example
  fixed linear optics), declared with its evidence. See
  {doc}`../practice/declarations`.

probe pixel
: The one detector pixel of the toy coronagraph whose brightness the theory
  pages follow as the source moves. See {doc}`../theory/toy-coronagraph`.

product
: One delivered image set with its own wavelength (or band) and
  normalization, the leading axis of the sample array.

PSF
: Point spread function: the image a point source (here a planet) makes on the
  detector, after the coronagraph.

quadrant grid
: A grid that stores only non-negative source positions, anchored at the
  origin, and reads negative positions through the declared two-axis mirror
  symmetry.

qualification read
: One of the reads compared with a direct propagation on
  {doc}`../validation`: the starter checks, positions near and just inside
  the inner working angle, and seeded random positions. Qualification reads
  are evidence, not bounds.

query
: The source position at which a value is wanted, between the samples. See
  {doc}`../foundations/sinc-interpolation`.

read
: Reconstruction of the PSF at one source position. A read returns an
  image and a status, and a refused read returns NaN images and its reasons.

reader
: The code that interpolates, an `Interpolator`: it takes a query, picks the
  stencil, weights its taps and sums them. See
  {doc}`../foundations/sinc-interpolation`.

reason bit
: One named flag in a refusal or read result, such as `KERNEL_MARGIN_FAIL`.
  The full table is on {doc}`../practice/admission`.

receipt
: The host-side record of an admission: identity, provenance, served
  region, symmetry and check residuals, allocation estimates. Readmitting
  changed data gives a new receipt identity.

refusal
: A decision not to return a value, recorded as reason bits. The yippy.sinc
  reader refuses rather than filling in a missing input or guessing a premise.
  See {doc}`../practice/admission`.

sample
: A value of the signal at a node of the grid. A sample is what a tap reads.

separable
: A two-dimensional weight that is the product $w(t_x)\,w(t_y)$ of two
  one-dimensional weights, so a read uses a $32\times32$ block of 1024
  taps. See {doc}`../theory/kernel`.

served region
: Source positions where a read is permitted: inside the declared domain and
  with all 32 by 32 stencil nodes present.

sinc
: The function $\operatorname{sinc}(t)=\sin(\pi t)/(\pi t)$: 1 at zero and 0 at
  every other integer, with a spectrum flat up to the Nyquist limit and zero
  beyond. See {doc}`../foundations/sinc-interpolation`.

source-dependent step
: Any processing step that changes with source position (recentering, a
  moving stamp); it breaks the band limit. See {doc}`../theory/band-limits`.

spacing rule
: The chain from support width $L$ to bandwidth $B=\sigma_{\max}L$, spacing
  $h\le0.76/(2B)$, node count $\lceil R/h\rceil+16$ per axis (where $R$ is the
  largest position the delivery serves along that axis) and the delivered grid.
  See {doc}`../theory/band-limits` and {doc}`../theory/quadrant-symmetry`.

spectrum
: The list of cosine frequencies and sizes that add up to a signal. See
  {doc}`../foundations/sampling`.

starter grid
: The public LUVOIR-A APLC delivery of 27 by 31 nodes at spacing 0.38
  $\lambda_0/D$. Its nodes give complete stencils anywhere in the mirrored
  4.18 by 5.70 box, and it declares only the corner cell
  $3.8\le s_x<4.18$, $5.32\le s_y<5.70$ as served. See {doc}`../validation`.

stencil
: The taps along one axis that a read uses around a query. For the raw profile
  these are 32 taps, from
  `floor(t) - 15` through `floor(t) + 16` with $t=(s-o)/h$: 16 at or below the
  query and 16 above. A two-dimensional read uses the $32\times32$ block of
  1024 taps formed by both axes. See {doc}`../foundations/sinc-interpolation`
  and {doc}`stencil-floor`. The prefiltered profile uses 5 to 10 taps per
  axis ({doc}`../practice/prefiltered-read`).

support width
: The full illuminated width $L$ of the pupil along an axis, in units of $D$.
  It fixes the bandwidth, $B=\sigma_{\max}L$.

tap
: One sample used by the reader, with its weight. For the raw profile the
  stencil is the set of 32 taps along an axis, and in two dimensions a read
  uses 1024 taps per pixel,
  each weighted by $w(t_x)\,w(t_y)$. See
  {doc}`../foundations/sinc-interpolation`.

tone
: A single cosine. The worst-tone error of a kernel is the largest error it
  makes on any one cosine inside the band, over every query offset in a cell.
  See {doc}`../theory/kernel`.

ULP
: Unit in the last place: the spacing between adjacent floating-point
  numbers at a given magnitude, about $2.2\times10^{-16}$ of the value in
  float64 and $1.2\times10^{-7}$ in float32.

tone error
: For the prefiltered profile, the largest error of a read on a cosine of
  unit amplitude inside the design band, over a sampled set of frequencies
  and query positions. The receipt records it; it is not an error bound.

unknown bound
: Version 0.1 reports every error bound as `+inf`. Reads in sampling mode
  still return images, and bounded requests refuse.

weight
: The multiplier $\operatorname{sinc}(t-n)$, or in yippy.sinc the Kaiser-sinc
  kernel at that distance, that a sample $n$ receives for the query $t$. A
  read is the sum of every tap's sample times its weight. Weights alternate in
  sign, so some samples count negatively. See
  {doc}`../foundations/sinc-interpolation`.

Next: {doc}`../viz`.

window
: A smooth taper multiplied into the cut sinc so its weights fall toward zero
  at the stencil edge instead of stopping abruptly. See {doc}`../theory/kernel`.

worst-tone error
: The largest error the kernel makes on a single cosine of size 1 at any
  frequency up to $\rho/2$ cycles per sample. See {doc}`../theory/kernel`.

YIP
: Yield input package: yippy's legacy directory of FITS and CSV files
  describing a coronagraph. See {doc}`../practice/yippy`.
