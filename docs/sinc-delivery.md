# Sinc delivery format

A sinc delivery is a directory of sampled off-axis planet PSFs that a coronagraph
designer propagates at positions chosen by the reader, plus a manifest that declares
what the samples mean. The reader is a 32-tap Kaiser-sinc kernel in `sinchole`, which is
loaded through yippy with `Coronagraph(path, interpolation="sinc")`. This page is the
complete contract for schema `sinc-delivery-0.1`. The parser in
`yippy.sinc_delivery` is the source of truth, and the notes below say which fields the
parser reads and which it only carries along.

Two commands help a designer meet the contract. Install the extra first.

```bash
pip install "yippy[sinc]"
yippy-sinc plan --lambda0 1e-6 --lambda-min 1e-6 --support 1 --limit 4.18 5.7 --out coords
yippy-sinc validate path/to/delivery
```

## Quickstart with the starter delivery

The starter delivery is a small public rehearsal: a physical-optics propagation of the
public LUVOIR-A APLC design on a 128 by 128 pixel frame. It holds 837 positive-quadrant
samples, 16 interior checks and 20 mirror checks, about 57 MB in total.

```python notest
import jax

from yippy.datasets import fetch_sinc_starter
from yippy.sinc_routing import route_interpolation

jax.config.update("jax_enable_x64", True)

path = fetch_sinc_starter()
_, backend, info = route_interpolation(path, "sinc", None, None)
psf = backend.create_psf(3.9425, 5.415)
```

The dataset URL is a placeholder until the release asset is published. The starter
holds no stellar or sky products, so `Coronagraph(path, interpolation="sinc")` does not
load it yet and the read goes through the router as shown. A full delivery that
includes the stellar products loads through `Coronagraph` directly. The quickest check
of the files is `yippy-sinc validate` on the unpacked directory.

## Choosing positions

The `plan` command applies three rules. Each is stated here so a designer can check the
numbers by hand.

- **Spacing.** The kernel needs a sampling occupancy of
  $\rho = 2\,(\lambda_0/\lambda_{\min})\,L\,h \le 0.76$ on each axis, where $L$ is the
  full illuminated entrance width over $D$ and $h$ is the node spacing in
  $\lambda_0/D$. The spacing is the largest $h$ that meets the limit, rounded down to
  six decimals. With $L=1$ and $\lambda_{\min}=\lambda_0$ this gives $h=0.38$.
- **Nodes.** A served limit $R$ (half-open, in $\lambda_0/D$) needs
  $\lceil R/h\rceil + 16$ stored nodes per axis, counted from index 0. The training
  list holds every node of that rectangle, both axes included, sorted by $y$ then $x$.
- **Checks.** Interior checks are 16 off-node positions by default, one in each of 16
  equal shells of the served radius, with a seeded angle so the set spreads around the
  quadrant. Mirror checks use eight base nodes, listed under the mirror-check columns
  below. The same seed always gives the same list.

One served limit means a circular region of that radius. Two limits mean a box
$[0,R_x) \times [0,R_y)$. The `plan.json` file written next to the lists gives the
`served` block and lattice values to copy into the manifest.

## Directory layout

```text
sinc-delivery.json                manifest (this page)
starter_training.csv              one row per stored node
starter_checks.csv                interior check positions
starter_mirror_checks.csv         negative-position check positions
mono-000/offax_psf.fits           float cube, (source, row, column)
mono-000/offax_psf_offset_list.fits   offsets, (source, 2)
checks/mono-000.fits              independent interior check images
mirror-checks/mono-000.fits       independent negative-position images
mirror-report.json                completed mirror and axis comparison
legacy-yip/                       optional, a separately identified legacy view
```

Every path in the manifest is relative to the delivery directory. File names other
than `sinc-delivery.json` are free, and the names above are the convention.

## Manifest keys

The manifest is one JSON object. "Required" means the parser raises or sinchole refuses
admission without it. Keys marked "carried" are stored in the receipt and hashed with
the manifest, but nothing reads their value.

### Top level

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `schema` | string | yes | Exactly `sinc-delivery-0.1`. |
| `layout` | string | yes | `quadrant_rectangle` (positive quadrant plus mirror) or `full_rectangle` (signed grid, `reflection` must be null). |
| `delivery_id`, `design_id` | string | no | Free labels, carried. |
| `lambda0_m` | number | yes | Reference wavelength in meters. The angular unit of every source coordinate is $\lambda_0/D$. |
| `diameter_m` | number | yes | Entrance diameter $D$ in meters. |
| `diameter_convention` | string | yes | Nonempty name of the diameter definition, for example `circumscribed_entrance`. |
| `coordinate_model` | string | yes | Exactly `paraxial_entrance_tilt`. |
| `source_unit` | string | yes | Exactly `lambda0/D`. |
| `offset_order` | string | yes | Exactly `xy`. |
| `array_order` | string | yes | Exactly `source,y,x`. |
| `source_origin_lod` | [number, number] | yes | Lattice origin. Exactly `[0, 0]` for a quadrant. |
| `source_spacing_lod` | [number, number] | yes | Node spacing $h_x, h_y$, each in $[10^{-6}, 10^{6}]$. |
| `offset_tolerance_rule` | string | yes | Exactly `four_ulp_pitch_cap`. |
| `support_extent_D` | [number, number] | for admission | Full illuminated entrance widths along x and y over $D$, finite and positive. |
| `support_evidence` | string | for admission | Nonempty text saying where the widths come from. |
| `premises` | object | for admission | The six premises, see below. |
| `kernel` | object | yes | Exactly `{"family": "kaiser_sinc_raw", "half_width": 16, "beta": 12, "max_occupancy": 0.76}`. |
| `served` | object | for admission | The permitted region, see Domain. |
| `reflection` | object or null | quadrant: yes, full: null | The symmetry declaration. |
| `detector_shape_yx` | [int, int] | yes | Rows, columns. Must be square and equal the FITS image shape. |
| `detector_center_xy` | [number, number] | yes | Zero-based column, row of the frame center. Exactly `((nx-1)/2, (ny-1)/2)`. |
| `detector_convention` | string | yes | Exactly `pixel_midpoint_v1`. |
| `pixel_scale_lod_xy` | [number, number] | yes | Pixel width in $\lambda_0/D$. Positive, square (both equal). |
| `pixel_scale_arcsec_xy` | [number, number] | yes | The same width in arcseconds, `lod * (lambda0_m / diameter_m) * 206264.80624709636`, at full precision. |
| `sample_kind` | string | yes | Pixel measure: `integral` or `mean`. See Pixel measure. |
| `pixel_response`, `instrument_to_sky` | string or null | no | Free descriptions, carried. |
| `normalization` | object | yes | See Normalization. |
| `optical_state` | object | yes | Must hold `state_id`. The whole object is hashed. Other fields are free, for example DM and wavefront hashes, alignment, polarization, coherence and propagator version. |
| `storage_dtype` | string | yes | `float32` or `float64`. Must equal the dtype of the PSF cube. |
| `accumulation_dtype` | string | no | Carried. |
| `spectral_sampling` | object | yes | `node_count` (equals the number of planes), `shortest_contributing_m` (equals the smallest plane value), `campaign_sampling_min_m` (number or null, the key must be present), optional `kind`, `band_edges_m`, `band_ready`. |
| `legacy_package_path` | string | for `Coronagraph` | Directory with the stellar and sky products. |
| `training_coordinates_path` | string | yes | The training CSV. |
| `planes` | list | yes | Wavelength planes, see Planes. |
| `mirror_checks` | list | quadrant: yes | Independent negative-position records. |
| `checks` | list | for pass | Independent interior check records. |
| `validation`, `stamp_validation` | object | no | Completed-report records. A record with `status` or `check_status` equal to `pass` or `fail` needs `report_path` and `report_sha256`. Any `report_path` or `refinement_report_path` is hash checked. |
| `extensions` | object | no | `file_sha256` maps each relative path to its SHA-256. Every entry is verified and the file must exist. Other keys are carried. |

The parser ignores unknown keys, so extra fields are safe, but they change the manifest
hash.

### Planes

Each entry of `planes` describes one monochromatic wavelength. Only one plane is read
at a time, chosen with `--product` or `SincOptions(product_id=...)`. With one plane no
choice is needed.

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `id` | string | yes | Unique nonempty identifier. |
| `kind` | string | yes | `monochromatic`. |
| `wavelength_m` | number | yes | Positive. Planes run from longest to shortest. |
| `band_edges_m` | [number, number] | yes | Both equal to `wavelength_m`. |
| `shortest_contributing_m` | number | yes | Equal to `wavelength_m`. |
| `sampling_min_wavelength_m` | number | for admission | The wavelength used to choose spacing, at most `wavelength_m`. |
| `psf_path`, `offsets_path` | string | yes | The cube and the offsets file. |
| `psf_sha256`, `offsets_sha256` | string | yes | SHA-256 of each file, lowercase hex. |
| `photon_conversion` | number | yes | Positive. Multiplies a stored value to give incident photon fractions. Exactly 1 for `incident_photon`. |
| `reference_uncertainty` | number or null | no | Fractional uncertainty of the normalization, at least 0. |

### Premises and evidence codes

`premises` has exactly these six names, each an object
`{"value": true, "evidence": 1, "source_id": "<text>"}`:

| Name | Meaning |
| --- | --- |
| `entrance_tilt_model` | Source position is a paraxial tilt across the entrance pupil. |
| `fixed_linear_optics` | The optics are fixed and linear in the field, whatever the source position. |
| `fixed_detector_registration` | The detector frame does not move between sources. |
| `fixed_pixel_response` | The pixel response is the same for every source. |
| `source_independent_normalization` | The normalization does not depend on source position. |
| `source_independent_spectral_weights` | Spectral weights do not depend on source position. |

Evidence codes are integers.

| Code | Meaning | Status |
| --- | --- | --- |
| 0 | None or unknown. | Admission refuses with `METADATA_MISSING`. |
| 1 | Declared by the designer, with a `source_id` naming the declaration. | Accepted. |
| 2 | Established by an external, verified source. | Rejected by the parser until verified external evidence is supported. |

A `value` of `false` is refused as a metadata conflict. The `source_id` must be a
nonempty string. `support_evidence` and `reflection.evidence` use the same idea as
text: a nonempty `support_evidence` counts as code 1, and `reflection.evidence` must be
the string `designer_declared`.

### Reflection

The quadrant declaration is an object. The parser checks every key listed.

| Key | Required value |
| --- | --- |
| `kind` | `two_axis_pixel_flip_v1`: reversing source x reverses detector columns and reversing source y reverses detector rows, with no other change to the image. |
| `x_mirror`, `y_mirror` | `true`. |
| `evidence` | `designer_declared`. |
| `declaration_id` | Nonempty identifier of the declaration. |
| `scope` | Nonempty text, the optical states and normalization it covers. |
| `response_argument_kind` | One of `component_reflections`, `mirror_plus_conjugation`, `other_response`. |
| `response_argument` | Nonempty text of the argument for the actual optical state. |
| `optical_state_id` | Equal to `optical_state.state_id`. |
| `check_tolerance_local_peak` | Exactly `1e-6`. |
| `check_status` | `pass`. |
| `report_path`, `report_sha256` | The completed mirror report and its hash. |

### Domain (the served region)

`served` states where the delivery permits reads.

| Key | Meaning |
| --- | --- |
| `kind` | `box` or `annulus`. |
| `boundary` | `closed`, or `lower_closed_upper_open`. Use the half-open form for a limit $R$. |
| `bounds_lod` | For `box`, `[[xmin, xmax], [ymin, ymax]]`. Otherwise null. |
| `radii_lod` | For `annulus`, `[inner, outer]` with `0 <= inner < outer`. Otherwise null. |

Every number has absolute value at most $10^{6}$ and is zero or a normal number. A
caller may restrict the domain with `SincOptions(domain=...)` but never widen it.
Independent checks must lie inside the served region and inside the stencil coverage.

### Pixel measure

`sample_kind` states what each stored pixel value is.

| Value | Meaning | `BUNIT` |
| --- | --- | --- |
| `integral` | Photon fraction integrated over the square pixel, per incident photon. | `incident_fraction_per_pixel` |
| `mean` | Mean photon fraction per $(\lambda_0/D)^2$ over the pixel. The reader multiplies by the pixel area. | `incident_fraction_per_lod2` |

Point samples are not supported. Never normalize an individual PSF to unit sum or to
its own peak.

### Normalization

`normalization.kind` is `incident_photon` (values are fractions of the photons incident
on the entrance, and every plane has `photon_conversion` equal to 1) or
`unocculted_peak` (each plane carries a positive `photon_conversion` to incident
photons). `source_independent` must be `true`. `included_losses` is text. The whole
object is hashed, so designers should also record the reference plane, collecting area,
quantum efficiency and uncertainty units there.

### Diameter convention

`diameter_m` is the diameter that defines $\lambda_0/D$. The convention name states
what it measures, and a circumscribed diameter of the entrance is the usual choice. The
FITS `D` keyword, `lambda0_m` and the arcsecond pixel scale must all agree with the
manifest.

## CSV files

The parser reads the training CSV. It does not open the check or mirror-check
lists. Their columns below are the convention that `yippy-sinc plan` writes, and the
manifest records are authoritative. The manifest records must agree with the lists.

### Training list

| Column | Type | Meaning |
| --- | --- | --- |
| `index` | int | Row of the FITS cube. A unique permutation of `0..N-1`. |
| `ix`, `iy` | int | Node indices, zero at the origin. A quadrant uses `0 <= ix` and `0 <= iy`. |
| `x_lod`, `y_lod` | float | `origin + spacing * index`, at full precision. |

The nodes must fill a complete rectangle with no duplicates or holes. Rows are sorted
by `iy` then `ix`, so `index = iy * nx + ix`.

### Check list

| Column | Type | Meaning |
| --- | --- | --- |
| `index` | int | Row of `checks/<plane>.fits`. |
| `id` | string | `check-000` and so on, matching the manifest record. |
| `x_lod`, `y_lod` | float | Off-node source position inside the served region. |

### Mirror-check list

| Column | Type | Meaning |
| --- | --- | --- |
| `index`, `id` | int, string | Row of `mirror-checks/<plane>.fits`, and `mirror-000` and so on. |
| `plane_id` | string | The plane this check belongs to. |
| `base_ix`, `base_iy` | int | The stored positive node the check mirrors. |
| `source_ix`, `source_iy` | int | Signed indices of the propagated source. |
| `x_lod`, `y_lod` | float | Signed source position, `source_index * spacing`. |
| `flip_x`, `flip_y` | 0 or 1 | Detector axis reversals applied to the base image. |

#### Why the starter has 20 mirror checks

The rule for a base with both indices nonzero is three counterparts, one for each
choice of negative sign: flip x only, flip y only, and both. A base on an axis has one
counterpart, because the other sign changes nothing. The starter uses eight bases:

| Base `(ix, iy)` | Kind | Counterparts |
| --- | --- | --- |
| (1, 1) | near the origin | 3 |
| (1, 14) | beside the y cell | 3 |
| (10, 1) | beside the x cell | 3 |
| (10, 14) | the served cell corner | 3 |
| (0, 14) | on the y axis | 1 |
| (10, 0) | on the x axis | 1 |
| (26, 1) | outer x node | 3 |
| (3, 26) | outer node row | 3 |

That is six bases with three counterparts and two axis bases with one, so
$6 \times 3 + 2 \times 1 = 20$. A description that says "three counterparts at each of
two bases" refers only to the two outer bases. The starter's last base uses row 26,
while the largest stored row is 30. The `plan` command uses the true outer row, so a
fresh plan for the same region ends with base (3, 30) and stays at 20 checks.

## FITS files

All FITS files have the image or table in the primary HDU alone.

### PSF cube

- **Array order.** `(source, row, column)`. FITS axes are `NAXIS1` columns,
  `NAXIS2` rows, `NAXIS3` sources, so `NAXIS=3`. Images are square. Increasing column
  follows positive source x and increasing row follows positive source y.
- **Dtype.** `BITPIX` of `-32` or `-64`, equal to `storage_dtype`. Values are finite and
  nonnegative.
- **Row order.** Row `k` is the node whose training `index` is `k`.

### Offsets file

- Shape `(source, 2)`, so `NAXIS=2` with `NAXIS1=2` and `NAXIS2=N`. Columns are x then y
  in $\lambda_0/D$.
- Float32 or float64, finite. Each value must match the lattice
  `origin + spacing * index` to four units in the last place of the largest offset, and
  that tolerance must stay below $10^{-4}$ of the spacing.

### Header keywords

Every PSF cube carries these keywords. The parser compares each with the manifest.

| Keyword | Value | Notes |
| --- | --- | --- |
| `LAMBDA` | `lambda0_m` | Wavelength in meters, within 8 ulp. |
| `MINLAM`, `MAXLAM` | the plane `wavelength_m` | Same unit rule as `LAMBDA`. |
| `D` | `diameter_m` | In meters. |
| `N_LAM` | 1 | Monochromatic product. |
| `PIXSCALE` | `pixel_scale_lod_xy[0]` | In $\lambda_0/D$ per pixel. |
| `XCENTER`, `YCENTER` | `detector_center_xy` | Zero-based, exactly equal. |
| `NAXIS1`, `NAXIS2` | detector columns, rows | Equal to `detector_shape_yx`. |
| `BUNIT` | `incident_fraction_per_pixel` or `incident_fraction_per_lod2` | Not read by the parser, but required by the request. |

The unit of `LAMBDA`, `MINLAM`, `MAXLAM` and `D` is read from the header comment (words
such as `m`, `nm` or `micron`). With no recognizable unit the parser assumes meters and
logs a warning, so write meters and put `m` in the comment. For `PIXSCALE` an empty
comment, `lambda0/D` or `lod` means $\lambda_0/D$ per pixel, and an angular unit such
as `arcsec` is converted. Any other keyword is ignored.

### Independent check and mirror images

Check and mirror cubes are float cubes of the same row and column shape, with finite
nonnegative values. Each manifest record names its file with `psf_path`, its row with
`image_index`, and the file hash with `psf_sha256`. The images are propagated directly
and never produced by flipping or interpolating delivered images.

## Check and mirror records

Interior check record keys:

| Key | Required | Meaning |
| --- | --- | --- |
| `id`, `plane_id` | yes | Identifier and the plane it belongs to. |
| `source_lod_xy` | yes | Finite position inside the served region and the stencil. |
| `psf_path`, `psf_sha256`, `image_index` | yes | The file, its hash and the row. |
| `independent_propagation` | yes | `true`. |
| `reference_units` | yes | `incident_fraction_per_pixel`. |
| `local_peak_reference` | for pass | Positive, finite. The largest direct pixel value among pixel centers within 1 $\lambda_0/D$ of the source. |
| `wavelength_m`, `stratum`, `aperture_flux_reference`, `peak_search_definition`, `aperture_definition` | no | Carried. |

Mirror record keys: `id`, `plane_id`, `image_index`, `base_index` (the training `index`
of the stored base node), `source_lod_xy` (signed), `psf_path`, `psf_sha256` and
`independent_propagation` equal to `true`. The sign of each coordinate selects which
detector axis is reversed on the base image.

## Mirror report

`mirror-report.json` is hashed by the parser (`reflection.report_sha256`) and the parser
requires it to exist. The comparison itself is recomputed from the images at admission,
so the report is a record for the designer and the receiver. The schema the request
tools write and read:

| Key | Type | Meaning |
| --- | --- | --- |
| `status` | string | `pass` or `fail`. |
| `threshold` | number | The limit, `1e-6`. |
| `mirror_rows` | list | One row per mirror check in list order: `id`, `base_index`, `pixel` (maximum absolute difference over the independent local peak), and optionally `aperture`, `peak`, `flux`, `minimum`. |
| `axis_rows` | list | One row per stored image with a zero index: `index` (training row), `axis` (0 for x, 1 for y), `pixel`, and the same optional fields. |
| `mirror_distribution`, `axis_distribution` | object | Optional summary of `pixel` (`min`, `median`, `p90`, `p95`, `max`). |
| `evidence`, `script` | string | Optional provenance. |

## Legacy view

`legacy-yip/` is optional. When present it holds a separately identified comparison
copy of the planet products in the older YIP conventions, with wavelength headers in
microns, and a short `README.txt` that says so. Its images and offsets equal the
delivered arrays. It is never used for sinc reads. `Coronagraph` reads stellar and sky
products from the directory named by `legacy_package_path`, so a complete YIP
package must live there when the object is constructed.

## Annotated example

The manifest below is complete and parses. Its geometry is deliberately tiny: a 4 by 4
detector and an 18 by 18 node rectangle, which is the smallest quadrant that satisfies
the 17-node minimum. A real delivery has the same keys with larger arrays. Text in
angle brackets stands for a 64-character SHA-256 digest that the designer replaces
with the hash of the named file. The test suite builds the matching small files and
parses exactly this block.

```javascript
{
  "schema": "sinc-delivery-0.1",
  "delivery_id": "example-small",
  "design_id": "example-design",
  "lambda0_m": 1e-06,                  // reference wavelength, meters
  "diameter_m": 15.0,                  // entrance diameter D, meters
  "diameter_convention": "circumscribed_entrance",
  "coordinate_model": "paraxial_entrance_tilt",
  "source_unit": "lambda0/D",
  "offset_order": "xy",
  "array_order": "source,y,x",
  "source_origin_lod": [0.0, 0.0],     // quadrant lattice starts at the origin
  "source_spacing_lod": [0.38, 0.38],  // h from rho = 2 (l0/lmin) L h <= 0.76
  "offset_tolerance_rule": "four_ulp_pitch_cap",
  "support_extent_D": [1.0, 1.0],      // entrance widths over D
  "support_evidence": "widths read from the entrance amplitude map",
  "premises": {
    "entrance_tilt_model": {"value": true, "evidence": 1, "source_id": "designer-declaration-1"},
    "fixed_linear_optics": {"value": true, "evidence": 1, "source_id": "designer-declaration-1"},
    "fixed_detector_registration": {"value": true, "evidence": 1, "source_id": "designer-declaration-1"},
    "fixed_pixel_response": {"value": true, "evidence": 1, "source_id": "designer-declaration-1"},
    "source_independent_normalization": {"value": true, "evidence": 1, "source_id": "designer-declaration-1"},
    "source_independent_spectral_weights": {"value": true, "evidence": 1, "source_id": "designer-declaration-1"}
  },
  "kernel": {"family": "kaiser_sinc_raw", "half_width": 16, "beta": 12, "max_occupancy": 0.76},
  "served": {                          // half-open box inside the 18-node coverage
    "kind": "box",
    "boundary": "lower_closed_upper_open",
    "bounds_lod": [[0.38, 0.76], [0.38, 0.76]],
    "radii_lod": null
  },
  "layout": "quadrant_rectangle",
  "reflection": {
    "kind": "two_axis_pixel_flip_v1",
    "x_mirror": true,
    "y_mirror": true,
    "evidence": "designer_declared",
    "declaration_id": "designer-declaration-1",
    "scope": "fixed optical state and normalization of this delivery",
    "response_argument_kind": "component_reflections",
    "response_argument": "the pupil, masks and detector are mirror symmetric about both axes",
    "optical_state_id": "example-state",
    "check_tolerance_local_peak": 1e-06,
    "check_status": "pass",
    "report_path": "mirror-report.json",
    "report_sha256": "<sha256 of mirror-report.json>"
  },
  "detector_shape_yx": [4, 4],
  "detector_center_xy": [1.5, 1.5],    // (n - 1) / 2 on each axis
  "detector_convention": "pixel_midpoint_v1",
  "pixel_scale_lod_xy": [0.25, 0.25],
  "pixel_scale_arcsec_xy": [0.0034377467707849396, 0.0034377467707849396],
  "sample_kind": "integral",           // photon fraction integrated over each pixel
  "pixel_response": "uniform square",
  "instrument_to_sky": null,
  "normalization": {
    "kind": "incident_photon",
    "source_independent": true,
    "included_losses": "apodizer, occulter and Lyot stop"
  },
  "optical_state": {"state_id": "example-state"},
  "storage_dtype": "float32",
  "spectral_sampling": {
    "node_count": 1,
    "shortest_contributing_m": 1e-06,
    "campaign_sampling_min_m": null
  },
  "legacy_package_path": "legacy",
  "training_coordinates_path": "starter_training.csv",
  "planes": [
    {
      "id": "mono-000",
      "kind": "monochromatic",
      "wavelength_m": 1e-06,
      "band_edges_m": [1e-06, 1e-06],
      "shortest_contributing_m": 1e-06,
      "sampling_min_wavelength_m": 1e-06,
      "psf_path": "mono-000/offax_psf.fits",
      "psf_sha256": "<sha256 of mono-000/offax_psf.fits>",
      "offsets_path": "mono-000/offax_psf_offset_list.fits",
      "offsets_sha256": "<sha256 of mono-000/offax_psf_offset_list.fits>",
      "photon_conversion": 1.0,
      "reference_uncertainty": null
    }
  ],
  "mirror_checks": [
    {"id": "mirror-000", "plane_id": "mono-000", "image_index": 0, "base_index": 19, "source_lod_xy": [-0.38, 0.38], "psf_path": "mirror-checks/mono-000.fits", "psf_sha256": "<sha256 of mirror-checks/mono-000.fits>", "independent_propagation": true},
    {"id": "mirror-001", "plane_id": "mono-000", "image_index": 1, "base_index": 19, "source_lod_xy": [0.38, -0.38], "psf_path": "mirror-checks/mono-000.fits", "psf_sha256": "<sha256 of mirror-checks/mono-000.fits>", "independent_propagation": true},
    {"id": "mirror-002", "plane_id": "mono-000", "image_index": 2, "base_index": 19, "source_lod_xy": [-0.38, -0.38], "psf_path": "mirror-checks/mono-000.fits", "psf_sha256": "<sha256 of mirror-checks/mono-000.fits>", "independent_propagation": true},
    {"id": "mirror-003", "plane_id": "mono-000", "image_index": 3, "base_index": 18, "source_lod_xy": [0.0, -0.38], "psf_path": "mirror-checks/mono-000.fits", "psf_sha256": "<sha256 of mirror-checks/mono-000.fits>", "independent_propagation": true},
    {"id": "mirror-004", "plane_id": "mono-000", "image_index": 4, "base_index": 1, "source_lod_xy": [-0.38, 0.0], "psf_path": "mirror-checks/mono-000.fits", "psf_sha256": "<sha256 of mirror-checks/mono-000.fits>", "independent_propagation": true}
  ],
  "checks": [
    {"id": "check-000", "plane_id": "mono-000", "image_index": 0, "source_lod_xy": [0.55, 0.62], "psf_path": "checks/mono-000.fits", "psf_sha256": "<sha256 of checks/mono-000.fits>", "independent_propagation": true, "reference_units": "incident_fraction_per_pixel", "local_peak_reference": 0.001302181621905289},
    {"id": "check-001", "plane_id": "mono-000", "image_index": 1, "source_lod_xy": [0.45, 0.7], "psf_path": "checks/mono-000.fits", "psf_sha256": "<sha256 of checks/mono-000.fits>", "independent_propagation": true, "reference_units": "incident_fraction_per_pixel", "local_peak_reference": 0.0012972177394635004}
  ],
  "extensions": {
    "file_sha256": {
      "mono-000/offax_psf.fits": "<sha256 of mono-000/offax_psf.fits>",
      "mono-000/offax_psf_offset_list.fits": "<sha256 of mono-000/offax_psf_offset_list.fits>",
      "checks/mono-000.fits": "<sha256 of checks/mono-000.fits>",
      "mirror-checks/mono-000.fits": "<sha256 of mirror-checks/mono-000.fits>",
      "starter_training.csv": "<sha256 of starter_training.csv>"
    }
  }
}
```

## Checking a delivery

`yippy-sinc validate <dir> [--product ID]` runs every integrity check above, then
the `sinchole` admission step with JAX binary64 on, and prints a pass or fail summary. The same
report is returned by `yippy.sinc_tools.validate`. It exits with 0 on a pass.

The summary shows the admission reason names and receipt id, $\rho$ on each axis, the
worst and median interior check residual against the acceptance level, and the mirror
and axis residuals.

- **Interior checks.** The residual is the largest absolute pixel difference between the
  read image and the independent image, divided by the independent local peak. The
  acceptance level is $10^{-5}$ of the local peak. The rehearsal delivery reads 1.4e-6
  at worst and 7.8e-7 at the median, so the level leaves a margin of about seven for a
  clean propagation. A tighter level of $10^{-6}$ would reject that clean delivery,
  and the blend reader on the same checks is off by about $0.2$.
- **Mirror and axis checks.** The limit is the fixed $10^{-6}$ of the declared
  tolerance. An axis image passes if its flip is bit-for-bit identical or its relative
  residual is within the limit.
- **Outer edge note.** The validator adds a note, without failing, when no mirror base
  reaches the largest stored index on an axis.

```bash
yippy-sinc validate path/to/delivery
yippy-sinc validate path/to/delivery --product mono-000 --level 1e-5 --json
```

```python notest
from yippy.sinc_tools import validate

report = validate("path/to/delivery")
print(report.summary())
assert report.passed
```

A pass is empirical evidence about the delivered finite model. It does not establish
physical accuracy, the correctness of the optical state, or an error bound between the
checks.
