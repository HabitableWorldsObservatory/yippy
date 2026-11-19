---
file_format: mystnb
kernelspec:
  name: python3
---

# What does each declaration claim?

A delivered grid is only readable between its nodes if a list of physical
statements about it is true, and most of those statements cannot be checked
from the images alone. The yippy.sinc package therefore asks for them as explicit
declarations, keeps each with its provenance in the receipt, and refuses
when one is missing rather than assuming it. This page goes through the
declarations in the order `admit_grid` takes them, saying what each claims
and why the reader needs it.

:::{admonition} The question of this page
:class: important

For each declaration: what does it claim, where does a designer write it, and
which refusal follows when it is missing or false? The close lists the answers.
:::

```{code-cell} python
:tags: [remove-cell]
import equinox as eqx
import hwostyle
import jax
import jax.numpy as jnp
import numpy as np
from myst_nb import glue

import yippy.sinc as sc
import toy_coronagraph as tc

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
```

## What do the declarations look like for the toy?

The toy coronagraph of the theory pages has the 96 by 96 pixel frame, the
0.25 $\lambda_0/D$ pixels and the other numbers listed in the box
{ref}`toy-in-numbers` on the first page of this section, and its helper
`tc.quadrant_delivery` fills in every declaration below. The cell builds that
delivery and prints each declaration on a labeled line, in the order of the
sections that follow, so each entry below can be matched to its line.

```{code-cell} python
from yippy.sinc.geometry import origin_xy, spacing_xy
toy = tc.ToyCoronagraph()
delivery = tc.quadrant_delivery(toy)
products, lattice = delivery["products"], delivery["declared_lattice"]
frame, pixels, norm = products.frame, products.pixels, products.normalization
reflection, domain, memory = delivery["reflection"], delivery["domain"], delivery["memory"]
nodes = delivery["intensity_nodes"]

print("intensity_nodes :", tuple(nodes.shape), nodes.dtype, "[products, nodes, rows, columns]")
print("offsets_xy_lod  :", tuple(delivery["offsets_xy_lod"].shape), "x then y, in lambda0/D")
print("declared_lattice: origin", origin_xy(lattice), "spacing", spacing_xy(lattice),
      "shape (ny, nx)", lattice.shape)
print("indices_xy      :", tuple(delivery["indices_xy"].shape), "integer (ix, iy) per node")
print("Frame           :", frame.reference_wavelength_nm, "nm,", frame.diameter_m, "m,",
      frame.diameter_convention, "|", frame.coordinate_model)
print("PixelGrid       :", pixels.shape, "center", jnp.asarray((pixels.center_x, pixels.center_y)),
      "pixel", jnp.asarray(pixels.basis)[0, 0], "lambda0/D,", products.measure)
print("Normalization   :", norm.reference, "| conversion", norm.photon_conversion,
      "| source", norm.provenance_id)
print("Wavelengths     :", products.kind, products.wavelength_nm, "nm, shortest",
      products.sampling_min_nm, "nm")
print("Support         : widths", products.support_extent_xy[0], "D, evidence",
      products.support_evidence, products.support_source_ids)
print("Premises        :", products.premises.values, "evidence", products.premises.evidence)
print("Identities      :", products.optical_id, products.spectral_weight_id, products.identity)
print("Reflection      :", reflection.kind, "axes", reflection.axes, "evidence",
      reflection.evidence, "tolerance", reflection.tolerance)
print("MirrorCheckSet  :", delivery["mirror_checks"].indices_xy.tolist(), "signed node indices")
print("Domain          :", domain.kind, domain.boundary, domain.limits_xy_lod.tolist())
print("MemoryBudget    :", memory.max_resident_bytes, memory.max_output_bytes,
      memory.max_scratch_bytes, memory.max_total_bytes, "bytes")
```

## Where does each declaration live in a delivery?

A designer never calls `admit_grid`. Each declaration is a key of the
`sinc-delivery.json` manifest or a column or HDU of a delivered file, and
yippy builds the call. The table has three columns: the yippy.sinc name (the
labels printed above), the yippy home that a designer writes, and what the
declaration states. The keys, columns and headers are defined on yippy's
[schema page][schema], which is the one place that defines them.

```{list-table}
:header-rows: 1
:widths: 20 42 38

* - yippy.sinc declaration
  - yippy key, column or file
  - What it states
* - `intensity_nodes`
  - the PSF cube of each plane, `planes[].psf_path`
    ([FITS files][schema-fits]), with `storage_dtype`
  - the images
* - `offsets_xy_lod`
  - the offsets file of each plane, `planes[].offsets_path`
  - the source position of each image
* - `declared_lattice`, `indices_xy`
  - `source_origin_lod` and `source_spacing_lod`, and the `ix`, `iy` columns of
    the training list named by `training_coordinates_path`
    ([CSV files][schema-csv])
  - the lattice the positions sit on and each image's integer node
* - `Frame`
  - `lambda0_m`, `diameter_m`, `diameter_convention`, `coordinate_model`
  - the reference wavelength and diameter that define $\lambda_0/D$
* - `PixelGrid`
  - `detector_shape_yx`, `detector_center_xy`, `pixel_scale_lod_xy`,
    `sample_kind` ([pixel measure][schema-pixel])
  - the detector geometry and what a pixel value measures
* - `Normalization`
  - the `normalization` object and `planes[].photon_conversion`
    ([normalization][schema-norm])
  - what one unit of intensity means
* - Wavelengths
  - `spectral_sampling` and `planes[]` ([planes][schema-planes])
  - the product wavelengths and the shortest one that contributes
* - Support
  - `support_extent_D` and `support_evidence`
  - the illuminated entrance width
* - `Premises`
  - `premises` ([premises and evidence codes][schema-premises])
  - six statements the band limit rests on
* - Identities
  - the optical identity is the hash of the `optical_state` object, and the
    product identity is the manifest hash joined to the plane `id`
  - names for the optical state, the weighting and the product
* - `Reflection`
  - `reflection` ([reflection][schema-reflection]) and the mirror report
  - the symmetry that lets one quadrant serve the field
* - `MirrorCheckSet`, `CheckSet`
  - `mirror_checks[]` and `checks[]` records and their cubes
  - images propagated directly, to test the symmetry and the reads
* - `Domain`
  - `served` ([domain][schema-domain])
  - where reads are permitted
* - `MemoryBudget`
  - not in the manifest, derived by yippy from the delivery unless the caller
    passes one
  - byte ceilings the reader may not exceed
```

A yippy delivery declares `point`, `integral` or `mean` for its detector samples.
A band-mean wavelength kind is not
accepted from a delivery, whose planes are monochromatic. And evidence code
2, external evidence, is rejected by the yippy parser, so a designer declares
each premise with code 1 and a `source_id` that names the declaration.

## The samples and their positions

`intensity_nodes`
: The images, a float32 or float64 array of shape `[P, N, Py, Px]`: products
  (for example wavelengths), source nodes, detector rows, detector columns.
  Values must be finite and non-negative. A negative zero is allowed, but a
  negative subnormal is not. Storage precision is kept as delivered.

`offsets_xy_lod`
: The source position of every node, shape `[N, 2]`, $x$ then $y$, in
  reference $\lambda_0/D$. Each must equal the node index times the declared
  spacing, plus the origin, to within four units in the last place (ULP) of
  the original precision, and that tolerance must also stay below $10^{-4}$ of
  the spacing. The offsets are never used to estimate the spacing. The original
  precision is the argument `offset_original_dtype` of `admit_grid`. It is an
  adapter argument and not a delivery field, because yippy sets it from the
  dtype of the offsets file.

`declared_lattice` (`optixstuff.grids.PixelGrid`)
: The authoritative origin, spacing and node count per axis. A designer
  states the spacing they propagated at, and yippy.sinc never fits it from
  rounded endpoints. The cell below shows why.

`indices_xy`
: The integer node index of every row of `intensity_nodes`, rebased so the
  smallest is zero. Rows may arrive in any order. Admission sorts them,
  rejects duplicates and requires the rectangle to be complete.

Node $i$ sits at $i\,h$, so an error $\delta h$ in the spacing moves node $i$
by $i\,\delta h$. Suppose the spacing were fitted from the last offset
written with six decimals. The cell compares the position error that allows,
at the last node of the starter, with the tolerance that admission applies to
float64 offsets and to float32 offsets.

```{code-cell} python
spacing, last_index = 0.38, 30
scale = {"x": 26 * spacing, "y": 30 * spacing}  # the largest stored offset on each axis
rounding = 0.5e-6  # half a unit of the sixth decimal
fit_error = rounding / last_index  # largest spacing error from a fitted endpoint
node_shift = last_index * fit_error
print(f"spacing fitted from an endpoint to six decimals: up to {fit_error:.1e} off,")
print(f"which moves node {last_index} by up to {node_shift:.1e} lambda0/D")
for dtype in (np.float64, np.float32):
    ulp = np.spacing(dtype(max(scale.values())))
    tau = 4 * float(ulp)
    print(f"{np.dtype(dtype).name} offsets: 4 ULP = {tau:.1e}, "
          f"bound 1e-4 * spacing = {1e-4 * spacing:.1e}, within bound: {tau < 1e-4 * spacing}")
glue("node_shift", f"{node_shift:.0e}", display=False)
glue("tau64", f"{4 * float(np.spacing(np.float64(max(scale.values())))):.0e}", display=False)
```

An endpoint rounded to six decimals could therefore move the last node by up
to {glue:text}`node_shift` $\lambda_0/D$, whereas admission accepts a
disagreement of only {glue:text}`tau64` for float64 offsets. The stated
spacing is the one the images were propagated at, so it is declared and used
as given.

## The physical product (`Products`)

`Frame`
: The reference wavelength $\lambda_0$ and diameter $D$ that define the unit
  $\lambda_0/D$, and which diameter convention $D$ refers to.

`PixelGrid`
: Detector shape, center in pixels, pixel basis in $\lambda_0/D$ (square and
  axis-aligned for quadrant grids). The product carries an optixstuff `Measure`.
  A delivery of centre samples decodes to `Measure.density("grid")`. Cell
  integrals and means decode to `Measure.integral()`. The reader preserves
  the declared measure. A quadrant
  needs its center at `((nx - 1)/2, (ny - 1)/2)`, and any other center is
  refused with `FRAME_UNSUPPORTED`.

`Normalization`
: What one unit of intensity means: the reference (for example incident
  photons at a declared plane), a conversion per product, an optional
  uncertainty and the factors included. Conversions must be the same for
  every source position, since a source-dependent factor breaks the band
  limit ({doc}`../theory/band-limits`).

Wavelengths
: `kind` (`monochromatic` or `band_mean`), the product wavelengths and band
  limits, the shortest wavelength that contributes to each product
  (`sampling_min_nm`) and an optional campaign minimum. The sampling test
  uses the shortest wavelength, because it sets $\sigma_{\max}$ and therefore
  the bandwidth.

Support
: The full illuminated entrance width along $x$ and $y$ in units of $D$
  (`support_extent_xy`), with an evidence code and a source identifier. The
  evidence code must be 1, designer declared, and any other value refuses
  admission with `UNKNOWN_BANDWIDTH`. Together with the shortest wavelength the
  width fixes $B=\sigma_{\max}L$ per axis, and admission requires
  $\rho=2Bh\le0.76$ on both axes.

`Premises`
: Six ordered statements that the band-limit argument depends on, numbered and
  explained in the next section.

Identities
: `optical_id`, `spectral_weight_id` and `identity` name the optical state,
  the spectral weighting and the product, and are hashed into the receipt.

(premises)=
## What are the six premises, and what happens when one fails?

The band limit of a pixel holds only if the optics behave in a particular
way, and yippy.sinc cannot test that from the images. The six premises state
it, in a fixed order, and each is declared true or false with an evidence code
and a source identifier.

1. The source enters as a paraxial tilt at the entrance pupil.
2. The optics after the tilt are fixed and linear.
3. The detector registration is fixed.
4. The pixel response is fixed.
5. The normalization does not depend on the source.
6. The spectral weights do not depend on the source.

The evidence code says who stands behind a premise. Code 1 means designer
declared and code 2 means externally established, and yippy.sinc accepts either
for a premise, but the yippy parser of version 0.1 rejects code 2, so a
delivery uses code 1 with a `source_id` that names where the statement is
argued. A code outside 1 and 2, such as 0 for unknown, refuses admission with
`METADATA_MISSING`. A premise declared false refuses with the reason in the
table, and through yippy a false premise raises a parse error before admission
is reached. The reader never assumes a premise.

```{list-table}
:header-rows: 1
:widths: 10 50 40

* - Premise
  - If declared false
  - Reason
* - 1 to 4
  - the tilt, the linear optics, the registration or the pixel response
  - `FRAME_UNSUPPORTED`
* - 5
  - the normalization depends on the source
  - `NORMALIZATION_UNSUPPORTED`
* - 6
  - the spectral weights depend on the source
  - `METADATA_CONFLICT`
```

The cell flips each premise of the toy in turn, then spoils one evidence code
and the support evidence, and prints the reasons that admission gives.

```{code-cell} python
def refusal(products):
    """Admit the toy delivery with replaced declarations and name the reasons."""
    changed = dict(delivery, products=products)
    admission = sc.admit_grid(**changed)
    return sc.reason_names(admission.reason_bits)

for index in range(6):
    false = eqx.tree_at(lambda p: p.premises.values, products,
                        products.premises.values.at[index].set(False))
    print(f"premise {index + 1} false:", refusal(false))
unknown = eqx.tree_at(lambda p: p.premises.evidence, products,
                      products.premises.evidence.at[0].set(0))
print("premise 1 evidence 0:", refusal(unknown))
external = eqx.tree_at(lambda p: p.support_evidence, products, jnp.array([2]))
print("support evidence 2:", refusal(external))
```

## Symmetry, permission and memory

`Reflection`
: `none` for a full signed grid, or `two_axis_pixel_flip_v1` for a quadrant,
  with the declaration, the argument that justifies it, its evidence code
  (1 or 2 for yippy.sinc), the check report and an optional stricter tolerance,
  never above $10^{-6}$ of the local peak. Admission computes the checks
  itself, and the `check_passed` field is ignored
  ({doc}`../theory/quadrant-symmetry`).

`MirrorCheckSet`
: For a quadrant grid, independent images at negative positions with their
  signed node indices: an $x$-negative, a $y$-negative and a both-negative
  counterpart for each product.

`CheckSet` (optional)
: Independent images at interior positions between nodes. Admission reads the
  grid at each and records the residual relative to the delivered
  `core_peak` as empirical evidence, never as a bound. The `core_peak` is the
  delivery's own number and admission does not recompute it.

`Domain`
: Where reads are permitted: a box or an origin-centered annulus, closed or
  lower-closed and upper-open. A read is served only inside the domain and
  where its whole stencil exists.

`MemoryBudget`
: Four ceilings, in bytes: `max_resident_bytes` for the stored grid,
  `max_output_bytes` for the images a call returns, `max_scratch_bytes` for
  working buffers and `max_total_bytes` for all of them together. The total
  also counts the two staging copies that admission makes of the grid.
  Admission refuses with `MEMORY_LIMIT` before allocating if an estimate
  exceeds a ceiling.

`provenance` (keyword)
: Optional `(key, text)` pairs, such as the full text of the symmetry
  argument or a normalization definition, kept verbatim in the receipt and
  hashed into its identity.

## What this page established

Each declaration is a claim that the images cannot prove, a key or file in the
delivery, and a named refusal when it is missing or false:

- Positions and lattice: the offsets sit on the declared lattice
  (`GRID_NONUNIFORM`), and the rectangle is complete (`GRID_INCOMPLETE`).
- Frame and pixel grid: one frame, with the quadrant center at
  `((nx - 1)/2, (ny - 1)/2)` (`FRAME_UNSUPPORTED`).
- Normalization: the same for every source (`NORMALIZATION_UNSUPPORTED`).
- Support and shortest wavelength: they fix the bandwidth
  (`UNKNOWN_BANDWIDTH`, `KERNEL_MARGIN_FAIL`).
- Premises: all six hold (`METADATA_MISSING`, `FRAME_UNSUPPORTED`,
  `NORMALIZATION_UNSUPPORTED`, `METADATA_CONFLICT`).
- Reflection and mirror checks: the symmetry is real
  (`SYMMETRY_CHECK_FAILED`, `SYMMETRY_CHECK_MISSING`).
- Domain and memory: reads stay where the stencil exists (`OUTSIDE_DOMAIN`,
  `INCOMPLETE_STENCIL`) and within the ceilings (`MEMORY_LIMIT`).

The next page orders the work of producing them.

Next: {doc}`delivery-recipe`.

[schema]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html
[schema-fits]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#fits-files
[schema-csv]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#csv-files
[schema-pixel]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#pixel-measure
[schema-norm]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#normalization
[schema-planes]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#planes
[schema-premises]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#premises-and-evidence-codes
[schema-reflection]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#reflection
[schema-domain]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#domain-the-served-region
