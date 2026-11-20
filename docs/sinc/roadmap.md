# Roadmap

The yippy.sinc package grows by adding readers, products and evidence beside the version
0.1 interfaces rather than by changing them. Each stage below says what it
unlocks for a user and what stays fixed, so code written against version 0.1
keeps working. The order reflects dependencies, not dates.

:::{admonition} Only version 0.1 exists
:class: warning

The stages Next, Then and Later are aspirational, apart from the release
hardening marked done under Next. They are directions the
design leaves room for, not commitments, and none has a date or an owner.
Version 0.1 is the one stage that is built.
:::

## Version 0.1: sampled intensity grids (now)

Version 0.1 reads off-axis planet intensity images from a regular grid of
propagated PSFs (the nodes of {doc}`foundations/sinc-interpolation`), either a
complete signed rectangle or a positive quadrant with a declared and checked
two-axis mirror symmetry. Admission checks the geometry, the declared physical
premises, the sampling limit and the memory budget. Every read either returns an
image with an unknown error bound or refuses with a reason. Reads run in JAX on
CPU, under `eqx.filter_jit` and `jax.vmap`, and `Coronagraph` uses it as a
sampled-intensity backend.

A second profile reads from 5 to 10 filtered images per axis instead of 32
delivered ones, on grids sampled at $\rho\le0.64$, and renders scenes of many
sources ({doc}`practice/prefiltered-read`). It carries the same evidence as
the raw profile, and it has been checked on synthetic grids only.

The starter delivery is a public LUVOIR-A APLC design. Its nodes support reads
anywhere in a mirrored 4.18 by 5.70 box, but its declaration serves only the
corner cell. Every independent check and every qualification read (the checks
are among the reads) lies in that cell, near the inner working angle, and the
counts are on {doc}`validation`. Elsewhere in the box, reads are supported but
unqualified.

## Next (aspirational): broader qualification

- **Release hardening (done).** The reader ships inside yippy, and continuous
  integration tests the lowest supported dependency versions.
- **More cells and designs.** Qualify additional cells of the APLC grid and
  other coronagraph families, such as hybrid Lyot and vortex designs, against
  independent propagations. The comparison method on {doc}`validation`
  carries over unchanged.
- **GPU execution.** The reader is fixed-shape JAX with arrays held as
  dynamic leaves, so it should run on a GPU without changes, but that has
  not been tested. Arithmetic, memory and speed need qualification there
  before a GPU result carries the same evidence as a CPU result.

## Then (aspirational): wavelength and stellar products

- **Wavelength-resolved products.** A spectral reader that combines several
  monochromatic products (the images delivered at one wavelength or band),
  using designer-chosen wavelength nodes and Chebyshev polynomials in inverse
  wavelength (a basis that represents smooth functions of wavelength with few
  nodes). Spatial reads stay exactly
  as they are, and the spectral layer calls them.
- **Band means.** A delivered band-mean image stays one exact product. Later
  versions can integrate wavelength-resolved products over a band (a
  weighted sum of reads, called quadrature) with an explicit error, but never split a band mean into channels.
- **Stellar products.** On-axis star images need integration over the
  stellar disk and pointing jitter, and sampling in stellar diameter. These
  arrive as separate readers with their own error targets, beside the
  off-axis reader.
- **Other layouts.** Sparse symmetric layouts that store only the useful
  annulus, radial grids and irregular grids each need their own reader and
  error model, behind the same abstract reader interface.

## Later (aspirational): error certificates and field tables

- **Error certificates.** Version 0.1 reports every error bound as unknown
  and refuses every bounded request. A certificate replaces "unknown" with
  an audited bound that holds under stated conditions (the declared premises):
  a ledger, meaning a list of every error term (truncation, storage precision,
  propagation convergence, symmetry), each with its evidence. Bounded reads then return an image with a stated maximum error,
  or refuse.
- **Complex field tables.** Storing the complex focal-plane field instead of
  intensity allows coherent sums, wavefront-error derivatives and a linked
  stellar wavefront layer. Fields carry a phase convention (a gauge, the
  arbitrary overall phase) that must be declared and checked, and are converted explicitly before any
  image is produced. Field tables are a separate type, not a subclass of the
  intensity grid, so intensity reads stay unchanged.
- **Scenes and consumers.** Weighted sums of many sources (disks, exozodi,
  multiple planets) with summed error terms, and direct use in
  exposure-time calculators and image simulators. The prefiltered profile
  already renders such sums ({doc}`practice/prefiltered-read`), with the
  error bound unknown.

## What stays fixed

Every stage above adds modules, readers and receipt fields. The version 0.1
names and signatures (`admit_grid`, `Interpolator`,
`read_batch`, `require_read`, the reason bits and statuses, all in
{doc}`/autoapi/index`) keep their meaning, existing
reads keep their numerical profile, and a missing input still refuses rather
than being filled in.

Next: {doc}`/autoapi/index`.
