---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  render_markdown_format: myst
---

# How we know: the physicaloptix starter delivery

The {doc}`theory/kernel` page measures the 32-tap Kaiser-sinc kernel's worst
error on a single cosine at about two parts in a million, as long as the
oversampling factor $\rho$ of {doc}`foundations/sampling` stays at or below
0.76. For a coronagraph that means a grid spacing of $h=0.38\,\lambda_0/D$
({doc}`theory/band-limits`). A prediction from a toy model is not evidence
about a real design, so the version-one reader was tested against a delivery
made the way a designer would make one: independent propagations of a real
coronagraph, compared with reads at positions the grid never saw. The question
for this page is how far the measured agreement, about one to two parts per
million of the planet's brightest nearby pixel, carries, and which denominator
it is measured against.

```{code-cell} python
:tags: [remove-cell]
import csv
import sys

sys.path.insert(0, "scripts")
from IPython.display import Markdown, display
from validation_tables import bin_tables, load, sci, static_dir

v = load("validation.json")
q = load("qualification_extended.json")
d = load("floor_diagnosis.json")
b = load("benchmark.json")
with open(static_dir() / "qualification_extended.csv") as handle:
    rows = list(csv.DictReader(handle))
radii = [float(r["r_lod"]) for r in rows]

# Design inputs of the starter delivery (see practice/declarations).
H = 0.38
DECLARED_L = (0.946, 0.982)  # declared illuminated widths along x and y, in D
NX, NY = 27, 31
rho_x, rho_y = (2 * width * H for width in DECLARED_L)
```

## The delivery

The design is the public LUVOIR-A apodized pupil Lyot coronagraph (APLC),
propagated with the open-source [physicaloptix](https://physicaloptix.readthedocs.io)
package at a single wavelength. The delivery follows {doc}`practice/designers`:

```{code-cell} python
:tags: [remove-input]
display(Markdown(
    f"- {v['training_nodes']} nodes (PSFs) on a {NX} by {NY} positive quadrant at spacing "
    f"{H} $\\lambda_0/D$, 128 by 128 pixels of 0.25 $\\lambda_0/D$, stored in float32.\n"
    f"- {v['independent_checks']} independent interior checks inside the corner cell "
    "$3.8\\le s_x<4.18$, $5.32\\le s_y<5.70$, propagated directly in float64. The "
    f"{NX} by {NY} nodes give complete mirrored stencils (the 32 by 32 taps of "
    "{doc}`foundations/sinc-interpolation`) for every $|s|$ below $(4.18, 5.70)$, so reads are "
    "supported anywhere in that mirrored box. This starter delivery declares only the corner cell as "
    "its served region, so reads elsewhere refuse for this delivery, and a delivery may declare "
    "the whole box. This cell is the one the checks cover, and it straddles the design inner "
    f"working angle of {q['design_iwa_lod']} $\\lambda_0/D$ (the separation at which the "
    "coronagraph is meant to start working).\n"
    f"- {v['mirror_checks']} independent mirror checks at negative positions. A stored base with "
    "both indices nonzero has three signed counterparts and a base on an axis has one, so "
    f"{v['mirror_bases_both_indices_nonzero']} bases times 3 plus "
    f"{v['mirror_bases_on_an_axis']} axis bases times 1 gives "
    f"{v['mirror_checks_at_those_bases'] + v['mirror_checks_at_axis_bases']} checks."
))
```

```{code-cell} python
:tags: [remove-input]
worst_local = q["all"]["sinc"]["local_peak"]["worst"]
display(Markdown(
    "The check positions were chosen before any read, and the comparison metric and the "
    "target of $10^{-5}$ of the local peak were fixed in advance. The target is the "
    f"accuracy the library was designed to meet, about {1e-5 / worst_local:.0f} times the worst "
    "local-peak error measured below. The `yippy-sinc validate` command on the starter "
    f"delivery reports PASS, with oversampling factors $\\rho$ of {rho_x:.4f} and "
    f"{rho_y:.4f} on the two axes against the limit of 0.76, and all {NX + NY} stored axis "
    f"images ({NX} along one axis and {NY} along the other) exactly invariant under "
    "reflection. The plan used a support width $L=1$, which gives $\\rho=2\\times1\\times"
    f"{H}={2 * H:.2f}$ on both axes. The declaration is smaller, {DECLARED_L[0]} by {DECLARED_L[1]} $D$ "
    "(a monochromatic delivery, so the reference wavelength is the shortest sampled one), "
    f"so $\\rho=2\\times{DECLARED_L[0]}\\times{H}={rho_x:.4f}$ along $x$ and "
    f"$2\\times{DECLARED_L[1]}\\times{H}={rho_y:.4f}$ along $y$, which leaves margins of "
    f"{0.76 - rho_x:.3f} and {0.76 - rho_y:.3f} under the limit."
))
```

## What the starter checks showed

Three metrics are reported for every read, and each one uses all 128 by 128
pixels of the image. The first is the one used on the landing page, and the
other two ask harder questions:

- Local peak: the largest absolute pixel difference between the read and the
  direct propagation, divided by the brightest direct pixel within
  1 $\lambda_0/D$ of the source.
- Pixel brightest: the largest over pixels of the absolute difference divided
  by that pixel's brightest stored value, taken over the quadrant and its
  three reflections. A pixel that is dim at every stored source position can
  have a large ratio for a tiny absolute error.
- Core flux: the relative error of the summed pixels whose centers lie within
  0.7 $\lambda_0/D$ of the true source, which is the number a yield code
  actually uses.

```{code-cell} python
:tags: [remove-input]
display(Markdown(
    "| Quantity | Result | Estimate or reference |\n|---|---|---|\n"
    f"| Admission | admitted, no reasons, {b['admission_first_call_seconds']:.0f} s on first call | none |\n"
    f"| Mirror checks, {v['mirror_checks']} positions | residual exactly {sci(v['mirror_residual_max'])} | "
    "exact symmetry of a real apodizer and Lyot stop |\n"
    f"| Axis invariance, all stored axis images | residual exactly {sci(v['axis_residual_max'])} | same |\n"
    f"| Worst of {v['independent_checks']} interior checks, error over local peak | "
    f"{sci(v['worst_check_error_local_peak'])} | kernel floor about "
    f"{sci(d['band']['kernel_tone_error_max_in_band'])} per tone, target $10^{{-5}}$ |\n"
    f"| Same worst check, error over each pixel's brightest stored value | "
    f"{sci(v['worst_check_error_pixel_brightest'])} | see the diagnosis below |\n"
    f"| yippy blend, worst of the same checks, error over local peak | "
    f"{sci(v['blend_worst_error_local_peak'])} | the dipole in the figure below suggests one planet core misplaced by a fraction of a node |"
))
```

The numbers come from `docs/scripts/validation_figures.py`, which reads the
delivery from its manifest, admits it with yippy.sinc and writes
`_static/validation/validation.json` beside the figures below.

```{figure} _static/validation/worst_check.png
:alt: Four panels around the worst interior check near 4 and 5.4 lambda0 over D. The direct propagation and the yippy.sinc read look identical. The yippy.sinc residual panel is blank at a color scale of one part in a million of the local peak. The blend residual shows a dipole across the planet core.
:width: 100%

Look at the two right panels, each on its own color scale: the labels give the
largest error over the local peak (about a millionth for yippy.sinc, a few
hundredths for the blend). The yippy.sinc residual has no structure at the planet
core, while the blend residual is a dipole there. The scales differ by several
orders of magnitude, so a residual of the blend's size would fill the
yippy.sinc panel with color.
```

```{figure} _static/validation/check_residuals.png
:alt: Three rails of residuals on a logarithmic axis. Independent checks lie between 3e-8 and 1.5e-6, left of the 1e-5 target. All mirror checks and all axis checks are exactly zero.
:width: 85%

Find where the independent checks sit against the $10^{-5}$ target on the top
rail. All of them fall below it, and every mirror and axis check is exactly
zero, drawn as a triangle at the left edge. The lower two rails use the
symmetry tolerance of $10^{-6}$ that admission enforces
({doc}`reference/numerical-contract`).
```

## Does the error hold between the checks?

The starter checks are few, and the positions a planet can take form a
continuum, so the evidence was widened before any claim was made. A seeded set of positions was drawn inside the
served cell and read by yippy.sinc and by yippy's blend interpolation, and each
read was compared with a direct float64 propagation.

```{code-cell} python
:tags: [remove-input]
g = q["groups"]
display(Markdown(
    f"- {g['starter']} starter checks, {g['behind']} positions just inside the design inner "
    f"working angle of {q['design_iwa_lod']} $\\lambda_0/D$, {g['near']} positions from that angle "
    f"to 0.2 above it, and {g['random']} uniform positions over the cell drawn with seed "
    f"{q['seed']}. That makes {q['all']['n']} reads, all in the declared corner cell, all near the "
    "inner working angle. Reads elsewhere in the supported box are not qualified.\n"
    f"- The reads span source radii from {min(radii):.2f} to {max(radii):.2f} "
    "$\\lambda_0/D$. The inner working angle here is the design value of the coronagraph, "
    "and a position just inside it is still far outside any occulter radius a toy model might "
    "use, so \"inside\" never means behind a hard occulter.\n"
    "- The three tables below are radius bins that re-sort the same reads by source radius. They "
    "are not the four draw groups above, and their counts differ from the group sizes.\n"
    f"- Photometric aperture radius {q['aperture_radius_lod']} $\\lambda_0/D$, and a local-peak radius "
    "of 1 $\\lambda_0/D$. Both are fixed values in the check scripts. The first three groups "
    "target the separations around the inner working angle, and the random group samples the "
    "cell uniformly."
))
```

A prediction before the tables: will the blend's median error over the local
peak, the same metric used for yippy.sinc, be near $10^{-3}$, near $10^{-1}$,
or near 1? The first figure on this page showed the blend's error at one position.

````{dropdown} Answer
The middle choice, a few percent to ten percent. The blend median over the
local peak is given in the second set of tables below and in the sentence after
the last figure. The blend gets the planet
core's brightness roughly right and its position wrong, so the error is a
fraction of the core and not the whole of it.
````

The blend is computed exactly as yippy's `OffJAX` does it in the version that
produced these numbers:

```{code-cell} python
:tags: [remove-input]
display(Markdown(
    f"- Version: yippy {q['yippy_version']}. Its routine reproduces the delivery's own stored "
    f"blend reads to {sci(q['blend_matches_delivery_reads_max_abs'])} in absolute value.\n"
    "- Take the four delivered images nearest the query by polar arc-length distance.\n"
    "- Weight each by $1/d^2$, with $d$ that distance, and normalize the weights to sum to one.\n"
    "- Shift each image to the query position with a Fourier phase ramp.\n"
    "- Sum the four shifted, weighted images."
))
```

### Sinchole reads

```{code-cell} python
:tags: [remove-input]
display(Markdown(bin_tables(q, "sinc", radii)))
```

### Blend interpolation, same positions and units

```{code-cell} python
:tags: [remove-input]
display(Markdown(bin_tables(q, "blend", radii)))
```

```{figure} _static/validation/qualification_radius.png
:alt: Three panels of relative read error against source radius, one per metric. The yippy.sinc points sit in a thin horizontal band from about 1e-7 to 1e-5 in every panel, and the blend points sit several orders of magnitude higher, with a spread of about a decade.
:width: 100%

Look at the vertical gap between the two clouds in each panel. The cyan
yippy.sinc reads stay near their floor across the gray line at the design inner
working angle, and the pink blend reads sit far above. The few points far
below each cloud are reads that lie almost on a node, explained in the text
below.
```

```{code-cell} python
:tags: [remove-input]
def node_distance(x, y):
    """Distance in lambda0/D from a position to the nearest node of the lattice."""
    dx = abs(x / H - round(x / H)) * H
    dy = abs(y / H - round(y / H)) * H
    return (dx**2 + dy**2) ** 0.5


on_node = [
    r for r in rows if node_distance(float(r["x_lod"]), float(r["y_lod"])) < 0.01 * H
]
display(Markdown(
    f"{len(on_node)} of the reads lie within {0.01 * H:.4f} $\\lambda_0/D$ (one hundredth of a "
    "node spacing) of a node. A read exactly on a node returns that node, so both methods are "
    "nearly exact there, and these are the lowest point of each cloud in the figure. The "
    "separation between the clouds does not depend on the metric or the radius, but it does "
    "depend on how far the read lies from a node."
))
```

```{code-cell} python
:tags: [remove-input]
s, a = q["all"]["sinc"], q["all"]["blend"]
display(Markdown(
    f"Over all {q['all']['n']} reads the yippy.sinc worst case over the local peak is "
    f"{sci(s['local_peak']['worst'])} and over each pixel's brightest value "
    f"{sci(s['pixel_brightest']['worst'])}, while the worst core flux error is "
    f"{sci(s['photometric']['worst'])}. The blend's worst values are "
    f"{sci(a['local_peak']['worst'])}, {sci(a['pixel_brightest']['worst'])} and "
    f"{sci(a['photometric']['worst'])}, with a median of {sci(a['local_peak']['median'])} over "
    "the local peak."
))
```

## Where does the pixel-relative error come from?

Measured against each pixel's brightest stored value, the worst error is
several times the local-peak error. Three candidates could explain it, and
each was measured separately.

```{code-cell} python
:tags: [remove-input]
st, bd = d["storage"], d["band"]
lines = bd["lines"]
edges = bd["declared_half_band"]
oob = max(p["out_of_band_rms_fraction"] for ln in lines.values() for p in ln["pixels"])
guard = max(ln["guard_lobe_halfwidth"] for ln in lines.values())
reach = {
    ax: max(p["highest_frequency_above_1e8_of_peak"] for p in ln["pixels"])
    for ax, ln in lines.items()
}
over = max(100 * (reach[ax] / edges[ax] - 1) for ax in reach)
cell_lv = [r for r in d["propagation_levels"] if r["position"].startswith(("worst", "node 1"))]
sw = d["kernel_on_exact_function"]["sweeps"]
kx, ky = sw["x"], sw["y"]
display(Markdown(
    "| Candidate | Measurement | Verdict |\n|---|---|---|\n"
    f"| Storage precision | Reading with float64 nodes changes the worst pixel error from "
    f"{sci(st['worst_pixel_relative_float32_nodes'], 4)} to "
    f"{sci(st['worst_pixel_relative_float64_nodes'], 4)}. float32 roundoff over the brightest value is "
    f"at most {sci(st['node_quantization_over_brightest_max'])}. | not the cause |\n"
    f"| A band wider than declared | The declared band edges are {edges['x']} along $x$ and "
    f"{edges['y']} along $y$ in cycles per $\\lambda_0/D$. Out-of-band energy beyond the edge plus the "
    f"estimator's guard lobe of {guard:.3f} is at most {sci(oob)} of the response rms at the worst pixels, "
    f"and the highest frequency above $10^{{-8}}$ of the peak sits at {reach['x']:.3f} and "
    f"{reach['y']:.3f}, at most {over:.0f} percent past the edges, inside the guard lobe of the "
    "windowed estimator. | not the cause: the overshoot is within the estimator's resolution and "
    "carries almost no energy |\n"
    f"| Propagator resolution | Two resolution levels of the same propagator (level 0 and level 2) differ by "
    f"{sci(min(r['level0_vs_level2_over_local_peak'] for r in cell_lv))} to "
    f"{sci(max(r['level0_vs_level2_over_local_peak'] for r in cell_lv))} of the "
    "local peak at the positions in the cell, a smooth model change far above the error being "
    "explained, and the same error appears when the kernel reads a finely sampled exact "
    "propagation. | not the cause |\n"
    f"| The kernel floor times each pixel's peak response | At the worst pixel by brightest value, "
    f"error over peak response is {sci(kx['worst_pixel_error_over_pixel_peak'])} along $x$ and "
    f"{sci(ky['worst_pixel_error_over_pixel_peak'])} along $y$, and over every pixel and query it "
    f"never exceeds {sci(max(kx['worst_error_over_pixel_peak_all_pixels_all_queries'], ky['worst_error_over_pixel_peak_all_pixels_all_queries']))}. "
    f"Error over the brightest stored value grows toward the cell edge. | consistent with the cause |"
))
```

The first row says storing the nodes in float64 leaves the error unchanged, so
it is not stored rounding. The third row needs one sentence: a different
resolution of the same propagator moves every pixel smoothly by far more than
$10^{-6}$, so a scatter of $10^{-6}$ in the nodes would be invisible against
that smooth change, and the finely sampled exact propagation shows the same
error without any such scatter.

```{figure} _static/validation/floor_diagnosis.png
:alt: Two panels of one-axis read error against source position, along x on the left and along y on the right. The cyan curve, error over each pixel's brightest stored value, is near 4 times 10^-7 below a source position of about 1.5, then climbs to about 6 times 10^-6 at the edge of the shaded cell. The pink curve, error over each pixel's peak response, stays flat near 4 times 10^-7 everywhere.
:width: 100%

Follow the cyan curve into the shaded cell (the declared served cell along each
axis) and then look at the pink curve beneath it. The cyan error climbs only where the pixel's stored values become
tiny compared with the response it would have at its own peak. The pink error,
which uses the pixel's peak response as denominator, does not move.
```

The proposed mechanism is one sentence long. A pixel far from the planet sees
almost nothing from sources inside the stored quadrant, so its brightest stored
value is small, but its response peaks at a source position where it is
large. The kernel's error scales with the response the stencil reads, which
is large at the peak, and not with the stored values, which are small. The
measurements are consistent with the kernel floor times each pixel's peak
response, and the error over that peak never exceeds the value in the last row
above.

```{code-cell} python
:tags: [remove-input]
wp = d["location"]["worst_pixels"][0]


def where(ax, sweep):
    inside = sweep["peak_inside_stored_lattice"]
    return (
        f"Along ${ax}$ the worst pixel by brightest value is ({sweep['worst_pixel_ix_iy'][0]}, "
        f"{sweep['worst_pixel_ix_iy'][1]}). Its peak response is "
        f"{sweep['pixel_peak_over_brightest_at_worst_pixel']:.0f} times its brightest stored value "
        f"and sits at source position {sweep['worst_pixel_peak_position_lod']:.2f} "
        f"$\\lambda_0/D$, which is {'inside' if inside else 'outside'} the stored lattice, "
        f"whose extent is {sweep['stored_lattice_extent_lod']:.2f}. The error over brightest is "
        f"{sci(sweep['worst_pixel_error_over_brightest'])} and over peak "
        f"{sci(sweep['worst_pixel_error_over_pixel_peak'])}."
    )


display(Markdown(
    where("x", kx) + "\n\n" + where("y", ky) + "\n\n"
    f"The diagnosis in the table above uses each sweep's own worst pixel. The single worst pixel of the "
    f"worst interior check is ({wp['ix']}, {wp['iy']}), {wp['r_from_source_lod']:.1f} "
    f"$\\lambda_0/D$ from the source, where the stored brightest is {wp['brightest_over_local_peak']:.2f} "
    "of the local peak. It is a different pixel, picked by a different criterion."
))
```

## Reading the numbers

```{code-cell} python
:tags: [remove-input]
floor = d["band"]["kernel_tone_error_max_in_band"]
peak_all = max(
    kx["worst_error_over_pixel_peak_all_pixels_all_queries"],
    ky["worst_error_over_pixel_peak_all_pixels_all_queries"],
)
display(Markdown(
    "| Denominator | Worst error over all reads | What it means |\n|---|---|---|\n"
    f"| Local peak | {sci(s['local_peak']['worst'])} | one to two parts per million of the planet core, "
    f"measured, against a kernel worst single tone of {sci(floor)} |\n"
    f"| Pixel's brightest stored value | {sci(s['pixel_brightest']['worst'])} | "
    "a few parts in a million, on pixels the quadrant barely lights |\n"
    f"| Pixel's own peak response | {sci(peak_all)} (one-axis sweeps) | "
    "the kernel floor times the pixel's response, flat in position |\n"
    f"| Core flux in a 0.7 aperture | {sci(s['photometric']['worst'])} | the quantity a yield code uses |"
))
```

```{code-cell} python
:tags: [remove-input]
display(Markdown(
    "So the headline, one to two parts per million, holds when the denominator is the local "
    "peak, the brightest pixel near the planet. It does not hold per pixel against the stored "
    f"brightest value, where the worst is {sci(s['pixel_brightest']['worst'])}."
))
```

Three different floors appear on these pages, and they should not be mixed.

```{code-cell} python
:tags: [remove-input]
display(Markdown(
    "- The kernel's worst single tone, measured on the {doc}`theory/kernel` page, is the "
    f"largest error one cosine in the band can suffer: {sci(floor)}.\n"
    "- The error over the local peak, measured over "
    f"the {q['all']['n']} reads, is {sci(s['local_peak']['worst'])}.\n"
    f"- The error over a pixel's peak response, measured above, is {sci(peak_all)}. It is "
    "lowest because a real response spreads its amplitude over many frequencies, and most of "
    "it sits well inside the band (a reading of the data, not measured separately).\n\n"
    f"The three lie within a factor of {floor / peak_all:.0f} of one another, ordered as "
    "peak response, local peak, tone. The kernel page does not predict this ordering, so it "
    "is an observation here."
))
```

The mirror checks are exactly zero because the physicaloptix propagation is
exactly symmetric under the declared reflections, which is what the
conjugation argument of {doc}`theory/quadrant-symmetry` predicts for a real
apodizer, a real Lyot stop and a hard occulting disk. This page does not
repeat the checks through yippy; {doc}`practice/yippy` shows that call.

## What this does not establish

- One cell, one wavelength, one design. All starter checks and all qualification
  reads lie in the declared corner cell, near the inner working angle. The nodes
  support reads anywhere in the mirrored box, but this delivery declares only
  that cell, and reads elsewhere are unqualified.
- Positions are seeded draws, not a proof over the continuum. The random
  positions sample the typical error. The worst case over those draws is
  stated above, and the worst case over the continuum can only be larger.
- The truth is a direct float64 propagation by the same code that made the
  nodes. It checks that yippy.sinc reproduces the model defined by the delivered
  arrays, and it does not test that model against a real telescope.
- Propagator resolution differs from level to level by far more than the read
  error, so the grid's own discretization of the optics, not the read, limits
  accuracy against the continuous design.
- Speed and memory are measured on one CPU only. See {doc}`reference/performance`.
  There is no GPU measurement ("GPU untested").
- Designs with complex focal-plane masks need their own symmetry evidence and
  checks.

Next: the {doc}`roadmap` lists the qualification still to come.
