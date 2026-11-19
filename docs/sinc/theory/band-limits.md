---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  render_markdown_format: myst
---

# What limits how fast a planet image changes?

A grid of images can be read between its nodes only if each pixel changes slowly
enough with the planet's position that the nodes pin it down. If a pixel could
change arbitrarily fast, no grid would be fine enough and every read would be a
guess. This page asks what sets the speed limit for a coronagraph, and turns the
answer into the node spacing a designer should use.

:::{admonition} What perfect interpolation needs
:class: important

1. **A band limit.** The signal has no cosine above a known bandwidth
   $f_{\max}$. **This page: the bandwidth of a coronagraph pixel, and the
   spacing it requires.**
2. **Dense enough samples.** $\rho=2f_{\max}<1$. *({doc}`../foundations/sampling`)*
3. **Weights a program can afford.** The Kaiser-sinc kernel, good to $\rho=0.76$.
   *({doc}`kernel`)*
:::

```{code-cell} python
:tags: [remove-cell]
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown
from myst_nb import glue

import eyepiece as ep
import toy_coronagraph as tc
from yippy.sinc import viz
from yippy.sinc.kernel import stencil, weights

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette
FAINT = ep.style.neutral(0.25)
SCENERY = ep.style.neutral(0.45)
NEUTRAL = ep.style.neutral(0.65)


def num(name, text):
    """Glue a plain string."""
    glue(name, text, display=False)


def tex(value, digits=1):
    """Format a number as a LaTeX power of ten."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return Markdown(rf"${mantissa}\times10^{{{int(exponent)}}}$")


PATH_Y = 0.5
STEP = 0.05
s_long = np.arange(-512, 512) * STEP


def probe_curve(toy, positions):
    """The probe pixel's intensity for a planet moving along x at s_y = 0.5."""
    x = np.asarray(toy.detector_x)
    col = int(np.argmin(np.abs(x - 3.1)))
    row = int(np.argmin(np.abs(x - 0.6)))
    images = np.asarray(toy.psf(np.stack([positions, np.full_like(positions, PATH_Y)], 1)))
    return images[:, row, col], images.max(axis=(1, 2))


def amplitude_spectrum(trace):
    """Hann-windowed amplitude spectrum, normalized to its maximum, with frequencies."""
    amp = np.abs(np.fft.fftshift(np.fft.fft(trace * np.hanning(trace.size))))
    freq = np.fft.fftshift(np.fft.fftfreq(trace.size, STEP))
    return freq, amp / amp.max()


def band_edge(trace, floor=1e-5):
    """Highest frequency whose normalized amplitude exceeds `floor`."""
    freq, amp = amplitude_spectrum(trace)
    return freq[freq > 0][amp[freq > 0] > floor].max()


toy = tc.ToyCoronagraph()
linear_trace, peak_value = probe_curve(toy, s_long)
s_fine = np.linspace(-6, 6, 961)
fine_trace, _ = probe_curve(toy, s_fine)
LOW = 1e-11
```

## How wide is the band of the probe pixel's curve?

The probe pixel's curve of the {doc}`toy coronagraph page <toy-coronagraph>`, the
brightness of one detector pixel as the planet moves, is the signal. The
property to look for is drawn the way the field draws it: the signal on the left,
its spectrum on the right, and the band shaded. A band-limited signal is one
whose spectrum drops to nothing to the right of a vertical line. The spectrum
here is computed from a longer record of the curve, $-25.6$ to $25.6$
$\lambda_0/D$ in steps of {glue:text}`b_step` $\lambda_0/D$, and its shaded band
ends at 1 cycle per $\lambda_0/D$.

```{code-cell} python
:tags: [hide-input]
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1, 1.1]})
left.semilogy(s_fine, fine_trace, color=pal.yellow, lw=2)
left.axvspan(-toy.occulter_lod, toy.occulter_lod, color=ep.style.neutral(0.15), lw=0)
left.set_ylim(LOW, 1)
left.set_xlabel(r"planet position $s_x$ ($\lambda_0/D$)")
left.set_ylabel("probe pixel intensity")
left.set_title("the signal", fontsize=10, color=SCENERY)
viz.plot_band_spectrum(linear_trace, STEP, bandwidth=1.0, ax=right)
right.set_xlim(-3, 3)
right.set_title("its spectrum", fontsize=10, color=SCENERY)
plt.show()

freq, amp = amplitude_spectrum(linear_trace)
inside = np.max(amp[(freq > 0.9) & (freq < 1.0)])
beyond = np.max(amp[(freq > 1.1) & (freq < 1.3)])
num("b_step", f"{STEP:g}")
num("b_edge", f"{band_edge(linear_trace):.2f}")
num("b_drop", f"{np.log10(inside / beyond):.0f}")
```

Look at the right panel: the spectrum falls steadily to the shaded edge and then
drops by about {glue:text}`b_drop` orders of magnitude between 0.9 and 1.1 cycles per
$\lambda_0/D$. The highest frequency whose amplitude stays above $10^{-5}$ of the
peak, a threshold chosen to sit above the window's leakage floor, is
{glue:text}`b_edge`. The slow decline beyond the edge belongs to the Hann
window used to compute the spectrum, not to the curve. So the curve behaves as
band-limited, with a bandwidth near $B=1$ cycle per $\lambda_0/D$ in planet
position. The next two sections show why that number is the width of the
entrance pupil.

## How does a planet position become a frequency?

A distant planet at source position $\mathbf{s}$ (a position on the sky in units
of $\lambda_0/D$, not the sample values $s(n)$ of the foundations pages) reaches
the entrance pupil as a tilted plane wave. The tilt multiplies the light at pupil
position $\mathbf{u}$, in units of $D$, by a **phase ramp**
$e^{2\pi i\sigma\,\mathbf{u}\cdot\mathbf{s}}$, where $\sigma=\lambda_0/\lambda$ is
1 at the reference wavelength. The strip shows the ramp for three planet positions
across the pupil: the farther the planet, the more phase cycles fit across it.

```{code-cell} python
:tags: [hide-input]
uu, vv = toy.pupil_coords
fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), constrained_layout=True)
for ax, sx in zip(axes, (0.5, 2.0, 4.0)):
    phase = np.angle(np.exp(1j * np.asarray(toy.phase_ramp((sx, 0.0)))))
    phase = np.where(np.asarray(toy.pupil) > 0, phase, np.nan)
    image = ax.imshow(phase, cmap=hwostyle.cmaps.phase, origin="lower",
                      extent=(-0.5, 0.5, -0.5, 0.5), interpolation="nearest",
                      vmin=-np.pi, vmax=np.pi)
    ax.set_title(rf"$s_x = {sx}\,\lambda_0/D$")
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([])
    ax.set_xlabel(r"pupil position $u_x$ ($D$)")
colorbar = fig.colorbar(image, ax=axes, label="phase of the ramp (radians)", shrink=0.8)
colorbar.set_ticks([-np.pi, 0, np.pi])
colorbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
plt.show()
```

The strip shows the ramp across space at one planet position. What matters for
the pixel is the other direction: how one pupil point's contribution changes as
the planet moves. At pupil position $u$ the contribution is a cosine in the
planet position $s$ with frequency $\sigma u$ cycles per $\lambda_0/D$, exactly the
kind of cosine that the foundations page added up to build curves. The figure
takes three points along $u_x$ at $\sigma=1$. Look at the three cosines: the
point at the centre does not oscillate at all, and the point at the edge of the
pupil, $u=0.5$, oscillates fastest.

```{code-cell} python
:tags: [hide-input]
s_axis = np.linspace(0, 4, 801)
points = ((0.0, NEUTRAL), (0.25, pal.green), (0.5, pal.purple))
fig = plt.figure(figsize=(9, 4.6), constrained_layout=True)
gs = fig.add_gridspec(3, 2, width_ratios=[1, 2.4])
pupil_ax = fig.add_subplot(gs[:, 0])
pupil_ax.add_patch(plt.Circle((0, 0), 0.5, fill=False, color=SCENERY, lw=1.5))
pupil_ax.set_xlim(-0.6, 0.6)
pupil_ax.set_ylim(-0.6, 0.6)
pupil_ax.set_aspect("equal")
pupil_ax.set_xlabel(r"pupil position $u_x$ ($D$)")
pupil_ax.set_yticks([])
pupil_ax.set_title("three points of the pupil", fontsize=10, color=SCENERY)
share = None
for k, (u, color) in enumerate(points):
    pupil_ax.plot(u, 0, "o", color=color, ms=9)
    ax = fig.add_subplot(gs[k, 1], sharex=share)
    share = ax if share is None else share
    ax.plot(s_axis, np.cos(2 * np.pi * u * s_axis), color=color, lw=2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([])
    ax.set_title(rf"point $u_x={u:g}$: frequency ${u:g}$ cycles per $\lambda_0/D$", fontsize=9,
                 color=color, loc="left")
    if k < 2:
        ax.tick_params(labelbottom=False)
axes_last = ax
axes_last.set_xlabel(r"planet position $s$ ($\lambda_0/D$)")
plt.show()
```

## Why does the intensity reach twice as far?

The detector does not record the light from one pupil point. It records the
intensity of the sum of every point's contribution, the sum squared. Take the two
edge points, $u=\pm0.5$. Their field, the sum of the two cosines, oscillates at
the edge frequency 0.5. Its square, the intensity, oscillates at twice that, 1,
because the product of the two edge contributions oscillates at the difference of
their frequencies, $0.5-(-0.5)=1$.

```{code-cell} python
:tags: [hide-input]
edge = 0.5
field = np.cos(2 * np.pi * edge * s_axis) * 2
intensity = field**2
fig, (a, b) = plt.subplots(2, 1, figsize=(9, 3.6), sharex=True, constrained_layout=True)
a.plot(s_axis, field, color=pal.purple, lw=2)
a.set_title(rf"field of the two edge points: ${edge:g}$ cycles per $\lambda_0/D$", fontsize=9,
            color=pal.purple, loc="left")
b.plot(s_axis, intensity, color=pal.yellow, lw=2)
b.set_title(rf"its square, the intensity: ${2 * edge:g}$ cycle per $\lambda_0/D$", fontsize=9,
            color=pal.yellow, loc="left")
for ax in (a, b):
    ax.set_yticks([])
b.set_xlabel(r"planet position $s$ ($\lambda_0/D$)")
plt.show()
```

Now the pupil as a whole. Every optic after the entrance pupil, the occulter and
the Lyot stop, is fixed and linear, so the field at detector position
$\mathbf{x}$ is

$$
E(\mathbf{x};\mathbf{s})=\int_P g(\mathbf{x},\mathbf{u})\,
e^{2\pi i\sigma\mathbf{u}\cdot\mathbf{s}}\,d\mathbf{u},
$$

where $P$ is the illuminated pupil and $g$ collects everything the fixed optics
do, whatever $\mathbf{s}$ is. Read as a function of $\mathbf{s}$, this is a sum of
cosines with frequencies $\sigma\mathbf{u}$, so the field has no frequency beyond
$\sigma L/2$, with $L$ the full width of the pupil in units of $D$. The intensity
$|E|^2=E\,E^*$ multiplies two such sums, so its frequencies are the differences
$\sigma(\mathbf{u}-\mathbf{u}')$ of pupil points, and along each axis its
bandwidth is

$$
B=\sigma_{\max}L .
$$

```{admonition} Predict before reading on
:class: tip
A design has a pupil 1.5 $D$ wide and works at the reference wavelength
($\sigma=1$). What is the highest frequency in a probe pixel's curve: 0.75, 1,
1.5 or 3 cycles per $\lambda_0/D$?
```

```{dropdown} Answer
1.5 cycles per $\lambda_0/D$: the two edge points of that pupil are $u=\pm0.75$
apart by 1.5, so $B=\sigma L=1.5$. The pupil here has $L=1$, and the measured edge,
{glue:text}`b_edge`, agrees.
```

## What does not set the band?

The occulter and the Lyot stop do not appear in $B$. They act after the tilt, and
nothing in $g$ depends on $\mathbf{s}$. The figure tests this on the toy by
changing each: the Lyot stop narrowed from {glue:text}`b_lyot_base` to
{glue:text}`b_lyot_new` $D$, and the occulter radius reduced from
{glue:text}`b_occ_base` to {glue:text}`b_occ_new` $\lambda_0/D$. The three
spectra share axes because the comparison is the point: the edge stays put.

```{code-cell} python
:tags: [hide-input]
variants = (
    ("baseline", tc.ToyCoronagraph(), pal.yellow, "-"),
    ("narrower Lyot stop", tc.ToyCoronagraph(lyot=0.5), pal.cyan, "--"),
    ("smaller occulter", tc.ToyCoronagraph(occulter_lod=1.5), pal.pink, ":"),
)
fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
edges = {}
for k, (name, variant, color, style) in enumerate(variants):
    trace, _ = probe_curve(variant, s_long)
    freq, amp = amplitude_spectrum(trace)
    keep = (freq >= 0) & (freq <= 2.5)
    ax.semilogy(freq[keep], np.maximum(amp[keep], 1e-12), color=color, lw=2, ls=style)
    edges[name] = band_edge(trace)
    ax.text(1.35, 0.25 / 4**k, name, color=color, fontsize=10)
ax.axvline(1.0, color=SCENERY, lw=1)
ax.set_ylim(1e-12, 3)
ax.set_xlim(0, 2.5)
ax.set_xlabel(r"frequency (cycles per $\lambda_0/D$)")
ax.set_ylabel("spectrum")
plt.show()
num("b_lyot_base", f"{toy.lyot:g}")
num("b_lyot_new", "0.5")
num("b_occ_base", f"{toy.occulter_lod:g}")
num("b_occ_new", "1.5")
num("b_edge_lyot", f"{edges['narrower Lyot stop']:.2f}")
num("b_edge_occ", f"{edges['smaller occulter']:.2f}")
```

Look at the three curves left of the vertical line at 1: they share one steep drop.
The edges, measured as before, are {glue:text}`b_edge` for the baseline,
{glue:text}`b_edge_lyot` with the narrower stop and {glue:text}`b_edge_occ` with the
smaller occulter. Changing them changed how bright the pixel gets, not how fast it
varies. Pixel integration is a different case, argued and not measured here: it
averages the intensity over a fixed footprint on the detector, which never touches
the source position, so it cannot add source frequencies.

## A near miss: a source-dependent step breaks the band

The argument above needs every step after the tilt to be the same for every
planet position. A **source-dependent step** breaks that. The figure repeats the
signal and spectrum of the first section on top, and below it the same curve after
one innocent-looking step: each image divided by its own peak value before the
pixel is read. The peak depends on where the planet is, so the division is not
fixed linear optics.

```{code-cell} python
:tags: [hide-input]
normalized = linear_trace / peak_value
fig, axes = plt.subplots(2, 2, figsize=(9, 5.4), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1, 1.1]})
for (left, right), trace, title in (
    (axes[0], linear_trace, "fixed linear optics"),
    (axes[1], normalized, "each image divided by its own peak"),
):
    window = np.abs(s_long) <= 6
    left.semilogy(s_long[window], np.maximum(trace[window], LOW), color=pal.yellow, lw=2)
    left.set_ylim(LOW, 1)
    left.set_title(title, fontsize=10, color=SCENERY)
    left.set_ylabel("probe pixel")
    viz.plot_band_spectrum(trace, STEP, bandwidth=1.0, ax=right)
    right.set_xlim(-4, 4)
axes[1][0].set_xlabel(r"planet position $s_x$ ($\lambda_0/D$)")
axes[0][0].set_xlabel("")
axes[0][1].set_xlabel("")
plt.show()

freq, amp_lin = amplitude_spectrum(linear_trace)
_, amp_norm = amplitude_spectrum(normalized)
far = (freq > 1.5) & (freq < 4)
glue("b_far_linear_tex", tex(amp_lin[far].max(), 0), display=False)
glue("b_far_norm_tex", tex(amp_norm[far].max(), 0), display=False)
```

Look at the right column. Above the edge the spectrum of the fixed-optics curve is
below {glue}`b_far_linear_tex` of its peak; after the division it stays near
{glue}`b_far_norm_tex`, many orders higher, because the divided curve has corners
where the brightest pixel changes. Corners are the same flaw as the triangle on
the foundations page, and they spill the spectrum past the band. So yippy.sinc
refuses products that apply a source-dependent step, such as dividing each image
by its own peak, recentering it, or using optics that change with field position.
Fixed factors are allowed: different products, for example different
wavelengths, may carry different fixed normalizations. A factor that varies with
the source may not. {doc}`../practice/admission` shows the refusal.

## How does a bandwidth become a node spacing?

The previous pages measured position in sample spacings; a grid has a spacing $h$ in
$\lambda_0/D$ between nodes. A cosine at $B$ cycles per $\lambda_0/D$ sampled every
$h$ has $f_{\max}=B\,h$ cycles per sample, the bandwidth of the
{doc}`sampling page <../foundations/sampling>`, so the oversampling factor is
$\rho=2f_{\max}=2Bh$. The {doc}`kernel <kernel>` holds its worst-tone error to
{glue}`b_floor` while $\rho\le0.76$, so the largest admissible spacing along an
axis is

$$
h\le\frac{0.76}{2B}=\frac{0.76}{2\,\sigma_{\max}L}.
$$

For a monochromatic product ($\sigma_{\max}=1$) and a pupil one diameter wide
($L=1$), $B=1$ and the Nyquist spacing is 0.5. The kernel allows $h=0.38\,
\lambda_0/D$, which puts the band at $0.38$ cycles per sample. That is the
bandwidth the foundations pages used, and it is why they chose it. A product
whose shortest contributing wavelength is $0.9\lambda_0$ has
$\sigma_{\max}={}$ {glue:text}`b_sigma` and needs {glue:text}`b_h_short`. A pupil
support wider than the diameter that defines $\lambda_0/D$ needs a proportionally
finer grid, which is why a delivery declares its full illuminated widths along $x$
and $y$ ({doc}`../practice/declarations`).

```{code-cell} python
:tags: [remove-cell]
sigma_short = 1 / 0.9
num("b_sigma", f"{sigma_short:.2f}")
num("b_h_short", f"{0.76 / (2 * sigma_short):.3f}")
```

Everything so far is a per-tone statement. The test that matters is the probe pixel
itself, read from a grid.

```{admonition} Predict before reading on
:class: tip
Read the probe curve from nodes spaced $h=0.5\,\lambda_0/D$, the Nyquist spacing
($\rho=1$), where the kernel page's worst tone is off by its whole amplitude. Is
the probe curve's worst error near its peak, near a tenth of it, or near
$10^{-4}$ of it?
```

The figure reads the curve from nodes along $s_x$ at three spacings, using the
library's kernel. The top axes show truth and read at $h=0.38$ on a logarithmic
axis. The bottom axes show the absolute error of each spacing. The dotted lines
are bounds: each cosine of the curve, times the kernel's worst-tone error at that
cosine's frequency, summed.

```{code-cell} python
:tags: [hide-input]
def read_curve(h, queries, origin=-12.0):
    """The probe curve read from nodes h apart, with the library's kernel."""
    nodes = origin + h * np.arange(int(24 / h) + 1)
    taps, w = stencil(jnp.asarray(queries), origin, h)
    values = probe_curve(toy, nodes)[0]
    return np.sum(np.asarray(w) * values[np.asarray(taps)], axis=1)


def tone_error(freq, rule, offsets=np.linspace(0.0, 1.0, 41)):
    """Worst error over query offsets, for a unit tone at each frequency."""
    tap = np.arange(-15, 17)
    w = rule(offsets[:, None] - tap)
    tone = np.exp(2j * np.pi * freq[:, None] * tap)
    truth = np.exp(2j * np.pi * freq[:, None] * offsets)
    return np.max(np.abs(tone @ w.T - truth), axis=1)


kernel_rule = lambda t: np.asarray(weights(jnp.asarray(t, dtype=jnp.float64)))
f_grid = np.linspace(0.0, 0.5, 501)
e_grid = tone_error(f_grid, kernel_rule)
cosines = np.abs(np.fft.rfft(linear_trace)) * 2 / linear_trace.size
cosines[0] /= 2
cos_freq = np.fft.rfftfreq(linear_trace.size, STEP)

queries = np.linspace(-4, 4, 801)
truth = probe_curve(toy, queries)[0]
spacings = ((0.25, NEUTRAL), (0.38, pal.cyan), (0.5, pal.pink))
fig, (top, bot) = plt.subplots(2, 1, figsize=(9, 6.4), constrained_layout=True,
                               gridspec_kw={"height_ratios": [1, 1.1]})
top.semilogy(queries, truth, color=pal.yellow, lw=2.2)
read_038 = read_curve(0.38, queries)
top.semilogy(queries, np.maximum(np.abs(read_038), 1e-14), color=pal.cyan, lw=1.2, ls="--")
top.set_ylim(1e-11, 1)
top.set_ylabel("probe pixel intensity")
top.set_title("truth (yellow) and the read from nodes 0.38 apart (dashed cyan)",
              fontsize=10, color=SCENERY)
results = {}
for h, color in spacings:
    err = np.abs(read_curve(h, queries) - truth)
    bound = float(np.sum(cosines * np.interp(cos_freq * h, f_grid, e_grid, right=1.0)))
    results[h] = (err.max(), bound)
    bot.semilogy(queries, np.maximum(err, 1e-14), color=color, lw=1.6)
    bot.axhline(bound, color=color, lw=1, ls=":")
bot.set_ylim(1e-11, 1e-2)
bot.set_xlabel(r"planet position $s_x$ ($\lambda_0/D$)")
bot.set_ylabel("absolute error of the read")
for (h, color), y in zip(spacings, (3e-3, 1e-3, 3.5e-4)):
    bot.text(-3.9, y, rf"$h={h}$, $\rho={2 * h:.2f}$", color=color, fontsize=10)
plt.show()

relative = np.abs(read_038 - truth) / truth
glue("b_floor", tex(e_grid[f_grid <= 0.38 + 1e-9].max(), 1), display=False)
at_45 = e_grid[np.argmin(np.abs(f_grid - 0.45))]
glue("b_w09", tex(at_45, 1), display=False)
num("b_w09_plain", f"{at_45:.2f}")
num("b_peak", f"{truth.max():.2f}")
glue("b_err_025", tex(results[0.25][0], 1), display=False)
glue("b_err_038", tex(results[0.38][0], 1), display=False)
glue("b_err_050", tex(results[0.5][0], 1), display=False)
glue("b_bound_038", tex(results[0.38][1], 1), display=False)
glue("b_bound_050", tex(results[0.5][1], 1), display=False)
num("b_err_050_peak", f"{results[0.5][0] / truth.max():.0e}".replace("e-0", "e-"))
glue("b_err_050_peak_tex", tex(results[0.5][0] / truth.max(), 0), display=False)
glue("b_err_038_peak_tex", tex(results[0.38][0] / truth.max(), 0), display=False)
num("b_rel_038", f"{relative.max():.0f}")
glue("b_truth_min", tex(truth.min(), 0), display=False)
```

Look at the dashed cyan line in the top panel. It follows the yellow curve through
the bump and the shallow dips, and departs from it only where the curve falls
below about $10^{-7}$, in the deep dips. At $h=0.38$ the worst absolute error is
{glue}`b_err_038`, which is {glue}`b_err_038_peak_tex` of the curve's peak,
{glue:text}`b_peak`. The error is a fraction of the signal's amplitude, not of its
value at the query, so where the curve is as low as {glue}`b_truth_min` the read
can be wrong by about {glue:text}`b_rel_038` times the local value. A
finer grid does not remove this: the $h=0.25$ error below is as large as the
$h=0.38$ error, so a guarantee on the faint parts, to a fraction of their own
value, is not available from this kernel.

```{dropdown} Answer
Near $10^{-4}$ of the peak: at $h=0.5$ the worst error is {glue}`b_err_050`, which is
{glue}`b_err_050_peak_tex` of the peak. The kernel page's worst tone is off by its
whole amplitude, but the curve has almost no amplitude near the band edge, as the
first spectrum shows, and the dotted bound of {glue}`b_bound_050`, the sum of each
cosine's amplitude times its tone error, lands on the measured error.
```

The dotted lines follow the same logic at the other spacings. At $h=0.38$ the bound
is {glue}`b_bound_038` and the error stays below it. At $h=0.25$ the error is
{glue}`b_err_025`, no smaller than at 0.38, because inside the margin the floor
belongs to the kernel and not to the grid.

## When does the bandwidth bound stop holding?

The bound holds for a planet that enters as a paraxial tilt at the entrance pupil
and passes through optics that do not change with its position: fixed masks, stops
and deformable mirror settings, a fixed pixel response, a fixed mixture of
polarizations summed incoherently, and spectral weights that do not depend on the
source. Pupil shear, field-dependent aberrations or source-dependent normalization
break it. These conditions are stated here, not tested; the near miss above tests
one of them. Within them the band limit is exact, but it says nothing about how
well the delivered images represent a real telescope. yippy.sinc interpolates the
model that the arrays define.

## Try it

1. A design's illuminated pupil is 1.1 $D$ wide along $x$ and 1.0 $D$ along $y$.
   Predict the largest admissible spacing along each axis for a monochromatic
   product, then check it with the formula.
   ```{dropdown} Hint
   $h_x\le0.76/2.2=0.345$ and $h_y\le0.38$. The axes may use different spacings,
   and admission checks each separately.
   ```
2. In the executed near-miss cell, replace the division by the peak with
   multiplication by a constant 2. Predict the spectrum before running it.
   ```{dropdown} Hint
   A constant does not depend on the source, so the spectrum keeps its shape and
   stays inside the band. Only its normalization changes, and the plot normalizes
   that away.
   ```
3. If a delivery accidentally used $h=0.45$ for $L=1$ at $\lambda_0$, what is
   $\rho$, and what worst-tone error should a read expect?
   ```{dropdown} Hint
   $\rho=2\times1\times0.45=0.9$, past the margin, where the worst-tone error is
   {glue}`b_w09`, about {glue:text}`b_w09_plain` of the amplitude. Admission refuses it
   ({doc}`../practice/admission` shows exactly this case).
   ```

## What this page established

:::{admonition} What perfect interpolation needs
:class: important

1. **A band limit.** A coronagraph pixel's brightness, as a planet moves, is
   band-limited with $B=\sigma_{\max}L$ cycles per $\lambda_0/D$: the width of the
   illuminated pupil, not the occulter, the Lyot stop or the detector.
2. **Dense enough samples.** $\rho=2Bh<1$, and the kernel's margin $\rho\le0.76$
   gives $h\le0.76/(2B)$, which is $0.38\,\lambda_0/D$ for a one-diameter pupil.
3. **Weights a program can afford.** The Kaiser-sinc kernel, good to the margin.
:::

A grid at the admitted spacing reads the probe pixel with a worst error of
{glue}`b_err_038`, a fraction of its amplitude and not of its faintest values. The
next page stores one quadrant of such a grid and reads the other three through
symmetry.

Next: {doc}`quadrant-symmetry`.
