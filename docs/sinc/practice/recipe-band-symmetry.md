---
file_format: mystnb
kernelspec:
  name: python3
---

# How do you deliver a filter, and which symmetry argument does the design support?

The positions, frame and units of the earlier pages assumed one wavelength and
one optical state. A real delivery has a filter with tails, and a design whose
optics decide whether one quadrant can stand for the whole field. This page
handles both, and each choice is something the library cannot make for you.

:::{admonition} The question of this page
:class: important

What shortest wavelength do you declare for a filter with tails, and which
design families may deliver only a quadrant? Step 4 answers the first and
step 5 the second. The close restates both.
:::

```{code-cell} python
:tags: [remove-cell]
import sys

sys.path.insert(0, "../scripts")
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown
from myst_nb import glue
from validation_tables import load

import yippy.sinc as sc
from yippy.sinc.evidence import reason_names
from yippy.sinc import viz
import toy_coronagraph as tc

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette


def tex(value, digits=2):
    """Format a number as inline math a x 10^b."""
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    return Markdown(rf"${mantissa}\times10^{{{int(exponent)}}}$")
```

(recipe-band)=
## Step 4: How do you deliver a filter with tails?

Version 0.1 delivers wavelengths as monochromatic planes. The `planes` list
runs from the longest to the shortest wavelength, each plane has its own
cube, offsets, checks and mirror checks, and one `--product` is read at a
time ([planes][schema-planes]). The lattice and the pixel scale are single
values in $\lambda_0/D$ for the whole delivery, so every plane is propagated
at the same angular positions on the same angular frame, and the
$\lambda_0/D$ grid is not rescaled with wavelength. The yippy parser of version
0.1 requires every plane to be monochromatic, so a band-mean plane is not
supported and a band product is assembled from the planes after the read.

When writing the assembled flat average as a native yield input package,
record its interval as `MINLAM < MAXLAM` in `offax_psf.fits` and its grid
wavelength as `LAMBDA`. Place this grid once per observing band at its centre
(Stark and Krist 2019, *Standardized Coronagraph Parameters for Input into Yield
Calculations*). A producer computing one wavelength must write equal limits.
A package with both limits absent also represents a single wavelength. The limits in
`stellar_intens.fits` must agree with the off-axis limits. The sky product
can have different limits in `sky_trans.fits`. The optional `N_LAM` sample
count serves only as a consistency check outside the specification.

The sampling limit is set by the shortest contributing wavelength, because
$\sigma_{\max}=\lambda_0/\lambda_{\min}$ sets the bandwidth. A filter with
tails has no sharp shortest wavelength, so declare it as the wavelength below
which the weighted flux fraction is less than a stated $\epsilon$, and write
$\epsilon$ in the optical state beside the filter curve. The library fixes
no value of $\epsilon$. A component beyond the edge is read at $\rho$ above
0.76, and its error is at most its weight times the worst single-tone kernel
error at that $\rho$. The figure plots that product for a tail carrying a
fraction $10^{-3}$ of the flux, against the interior acceptance level of
{glue}`level` of the local peak that {doc}`recipe-check-send` recommends. Look at
where the curve crosses the dashed line.

```{code-cell} python
:tags: [hide-input]
weight = 1e-3
rhos = np.array([0.76, 0.80, 0.85, 0.90, 0.95, 1.00])
worst = np.array([tc.kernel_error_worst(float(rho)) for rho in rhos])
tail = weight * worst
for rho, w, t in zip(rhos, worst, tail):
    print(f"rho = {rho:.2f}: worst single-tone error {w:.1e}, a 1e-3 flux tail contributes {t:.0e}")
fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
ax.semilogy(rhos, tail, marker="o", color=pal.cyan)
ax.axhline(1e-5, color="0.6", linestyle="--")
ax.text(0.76, 1.3e-5, "interior level", color="0.8", fontsize=9)
ax.set_xlabel(r"$\rho$ of the tail component")
ax.set_ylabel("error contributed by the tail")
plt.show()
above = rhos[tail > 1e-5]
glue("level", tex(1e-5, 1), display=False)
glue("first_above", f"{above.min():.2f}", display=False)
glue("last_below", f"{rhos[tail <= 1e-5].max():.2f}", display=False)
glue("tail_rho1", tex(float(tail[-1])), display=False)
glue("worst_rho1", f"{worst[-1]:.2f}", display=False)
```

The tail stays below the level up to $\rho=$ {glue:text}`last_below` and first
exceeds it at $\rho=$ {glue:text}`first_above`. At $\rho=1$ the worst single-tone error is
{glue:text}`worst_rho1` of the tone, so the tail contributes about its whole
weight, {glue}`tail_rho1`. The estimate is rough, so let the interior checks
decide. The per-plane `sampling_min_wavelength_m` must not exceed the plane
wavelength, and `campaign_sampling_min_m` records a single minimum for the whole
campaign.

(recipe-symmetry)=
## Step 5: Which symmetry argument does the design support?

The quadrant needs both intensity mirrors for the optical state actually
propagated. The arguments are on {doc}`../theory/quadrant-symmetry`. Two
geometric mirrors of every component give both directly. Conjugation,
$E(-x;-s)=E^*(x;s)$, which says that the field at the opposite detector point
for the opposite source is the complex conjugate of the original, needs a real
pupil, a real stop and a real centrosymmetric focal mask, and combined with one
geometric mirror it gives both. A complex mask or a wavefront aberration breaks
conjugation. The table applies this to the common families.

```{list-table}
:header-rows: 1
:widths: 20 45 35

* - Design family
  - Symmetry argument
  - Layout
* - APLC, real occulter, pupil with two mirrors
  - Two geometric mirrors (`component_reflections`), and conjugation as a
    second route
  - Quadrant: `two_axis_pixel_flip_v1`, nodes $\lceil R/h\rceil+16$ per axis
* - APLC, real occulter, pupil with one mirror (the public starter)
  - One mirror plus conjugation (`mirror_plus_conjugation`), valid only while
    the optics stay real
  - Quadrant, as above
* - HLC, complex focal mask
  - Conjugation fails. A quadrant needs two geometric mirrors of the pupil and
    a mask phase that is itself mirror symmetric
  - Quadrant if both hold, otherwise full grid
* - Vortex
  - Mirrors reverse the charge, see the handedness note below
  - Quadrant for a two-mirror pupil, full grid otherwise, checked per
    polarization channel
* - Single-mirror pupil with a complex mask, or a segmented or off-axis pupil
    with no mirror
  - None gives both mirrors. With a complex mask, conjugation fails even for
    the point inversion $I(-x;-s)=I(x;s)$, as the both-negative residual in the
    toy cell below shows. Version 0.1 uses neither
  - Full grid
```

The full-grid recipe replaces the quadrant steps. Set `layout` to
`full_rectangle` and `reflection` to null, deliver no mirror checks, and list
signed indices in the training file with the origin at $(0,0)$. A served box
$[-R,R)$ on each axis then needs indices from $-(\lceil R/h\rceil+15)$ to
$+(\lceil R/h\rceil+15)$, which is $2\lceil R/h\rceil+31$ nodes per axis. The cell
counts them for the starter region. The interior checks and everything in
steps 2 to 4 are unchanged.

```{code-cell} python
v = load("validation.json")
cells = np.ceil(np.array([4.18, 5.7]) / 0.38).astype(int)
full = 2 * cells + 31
print(f"full grid: {full[0]} x {full[1]} = {full[0] * full[1]} nodes; "
      f"quadrant: {v['training_nodes']}")
glue("full_nodes", f"{full[0]} by {full[1]}, {full[0] * full[1]}", display=False)
glue("quad_nodes", str(v["training_nodes"]), display=False)
```

That is a {glue:text}`full_nodes` node grid against {glue:text}`quad_nodes`, about four times
the quadrant. The planner writes quadrant lists only, so write the full-grid
training list by hand from the rule above, and `yippy-sinc validate` confirms
the node count.

### Why does a vortex need a symmetric pupil?

A vortex of charge $\ell$ has mask phase $e^{i\ell\theta}$, and reflecting the
system about an axis maps $\theta$ to $-\theta$, so the reflected optical system
carries charge $-\ell$. The mirror image of a charge $\ell$ product is a charge
$-\ell$ product. When the pupil has both mirrors the two products have the same
intensity, because conjugation maps $+\ell$ to $-\ell$ for a real pupil and a real
stop, and the quadrant holds. When the pupil has one mirror, that geometric
mirror maps one handedness to the other and the check fails.

The two circular polarizations see opposite charges, so polarization channels
are checked one channel at a time, and an unpolarized sum can be symmetric while
each channel is not. Version 0.1 has no polarization axis in the schema.

### Does a complex mask really break the quadrant?

The claim is measured on the toy coronagraph. The cell gives the toy's focal
mask a phase of 0.8 radians that grows as the square of the radius, a
centrosymmetric but complex mask, and a vortex phase of charge 2, and admits a
quadrant grid for each of five cases. The toy pupil that is shifted along $x$
keeps one mirror and is the one-mirror case.

```{code-cell} python
:tags: [hide-input]
from dataclasses import dataclass

@dataclass(frozen=True)
class ComplexMaskToy(tc.ToyCoronagraph):
    """The toy with a complex focal mask: radial phase and an optional vortex charge."""

    mask_phase: float = 0.0
    charge: int = 0

    def field(self, source_xy):
        uu, vv = self.pupil_coords
        u = uu[0]
        stop = (uu**2 + vv**2 <= (self.lyot / 2) ** 2).astype(jnp.float64)
        n_occ = 40
        xi = tc._axis(n_occ, 2 * self.occulter_lod / n_occ)
        xx, yy = jnp.meshgrid(xi, xi)
        r2 = (xx**2 + yy**2) / self.occulter_lod**2
        disk = (r2 <= 1).astype(jnp.float64)
        mask = disk * jnp.exp(1j * self.mask_phase * r2)
        if self.charge:
            mask = disk * (1 - jnp.exp(1j * self.charge * jnp.arctan2(yy, xx)))
        du = 1.0 / self.n_pupil
        to_focal = jnp.exp(-2j * jnp.pi * jnp.outer(xi, u)) * du
        to_lyot = jnp.exp(2j * jnp.pi * jnp.outer(u, xi)) * (xi[1] - xi[0])
        to_detector = jnp.exp(-2j * jnp.pi * jnp.outer(self.detector_x, u)) * du
        tilted = self.pupil * jnp.exp(1j * self.phase_ramp(source_xy))
        focal = to_focal @ tilted @ to_focal.T
        lyot_field = tilted - to_lyot @ (mask * focal) @ to_lyot.T
        return to_detector @ (stop * lyot_field) @ to_detector.T

cases = {
    "one mirror, real mask": ComplexMaskToy(pupil_shift=(0.02, 0.0)),
    "one mirror, complex radial mask": ComplexMaskToy(pupil_shift=(0.02, 0.0), mask_phase=0.8),
    "two mirrors, complex radial mask": ComplexMaskToy(mask_phase=0.8),
    "two mirrors, vortex phase (charge 2)": ComplexMaskToy(charge=2),
    "one mirror, vortex phase (charge 2)": ComplexMaskToy(pupil_shift=(0.02, 0.0), charge=2),
}
receipts, reasons = {}, {}
for name, case in cases.items():
    admission = sc.admit_grid(**tc.quadrant_delivery(case))
    receipts[name] = admission.receipt
    reasons[name] = reason_names(admission.reason_bits)
    print(f"{name}: admitted={admission.admitted}, reasons={reasons[name]}")
    print("   mirror residuals (x-neg, y-neg, both-neg):",
          ", ".join(f"{v:.1e}" for v in admission.receipt.mirror_local_peak))
```

What to look at in the figure: each panel is one case, and the dots on the
middle rail are its three mirror residuals against the dashed limit. In the
three admitted cases all three dots sit at rounding level. In the two refused
cases two of the three dots sit far above the limit, and the third sits at
rounding level.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, constrained_layout=True)
for k, (ax, (name, receipt)) in enumerate(zip(axes.ravel(), receipts.items())):
    viz.plot_check_residuals(receipt, ax=ax)
    ax.set_title(name, fontsize=10)
    if k % 3:
        ax.tick_params(labelleft=False)
axes.ravel()[-1].axis("off")
plt.show()
quantities = {name: np.asarray(r.mirror_local_peak) for name, r in receipts.items()}
complex_one = quantities["one mirror, complex radial mask"]
vortex_one = quantities["one mirror, vortex phase (charge 2)"]
glue("cx_big", f"{complex_one[0]:.2f}", display=False)
glue("cx_small", tex(float(complex_one[1])), display=False)
glue("vx_big", f"{vortex_one[1]:.3f}", display=False)
glue("vx_small", tex(float(vortex_one[0])), display=False)
```

The one-mirror pupil with a real mask passes at rounding level, as the theory
predicts. With the complex mask the same pupil is refused with
`SYMMETRY_CHECK_FAILED`. The $x$-negative and both-negative residuals are
{glue:text}`cx_big` of the local peak and the $y$-negative residual, the
pupil's own mirror, is still {glue}`cx_small`, so the failure shows which
reversal conjugation was supplying. The two-mirror pupil with the same complex
mask passes, because two geometric mirrors do not use conjugation.

The toy vortex is an azimuthal phase of charge 2 confined to the occulter
disk, so it is a test of the handedness argument and not a vortex design.
Behind the two-mirror pupil it passes. Behind the one-mirror pupil it is
refused, with the $y$-negative and both-negative residuals at
{glue:text}`vx_big`. The geometric mirror of that pupil is the one that reverses
the charge, and the $x$-negative check, which conjugation supplies, still passes
at {glue}`vx_small`.

## What this page established

A filter with tails is declared by the wavelength below which the weighted flux
is under a stated $\epsilon$, and a component below it is judged by the
interior checks. A quadrant is only for designs whose real optics or two
mirrors give both intensity mirrors, and the toy showed a complex mask failing
the check where a real one passes. The last page checks the files and sends
them.

Next: {doc}`recipe-check-send`.

[schema-planes]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#planes
