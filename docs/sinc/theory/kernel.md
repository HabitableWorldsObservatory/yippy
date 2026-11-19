---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  render_markdown_format: myst
---

# How close to the Nyquist limit can a 32-tap kernel go?

{doc}`The previous page <../foundations/sinc-interpolation>` ended with a
problem. The ideal sinc needs every sample, a program can afford 32 taps per
axis, and cutting the sinc at the edge of the stencil leaves an error of
several percent of the signal. This page asks how a smooth taper repairs the
cut, and how close to the Nyquist limit a grid may then go before the repair
stops working.

:::{admonition} What perfect interpolation needs
:class: important

1. **A band limit.** The signal has no cosine above a known bandwidth
   $f_{\max}$. *({doc}`../foundations/sampling`; a coronagraph's bandwidth
   is found in {doc}`band-limits`.)*
2. **Dense enough samples.** $\rho=2f_{\max}<1$.
   *({doc}`../foundations/sampling`)*
3. **A way to combine the samples that a program can afford.** Weights that
   are a sinc, kept to 32 taps without leaving a large error. **This page.**
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
from yippy.sinc.kernel import weights

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette
FAINT = ep.style.neutral(0.25)
SCENERY = ep.style.neutral(0.45)
KEPT = ep.style.neutral(0.65)
CUT = ep.style.neutral(0.55)

W_HALF = 16
BETA = 12.0


def tex(value, digits=1):
    """Format a number as a LaTeX power of ten."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return Markdown(rf"${mantissa}\times10^{{{int(exponent)}}}$")


def num(name, text):
    """Glue a plain string."""
    glue(name, text, display=False)


def kaiser(t, beta=BETA):
    """The Kaiser window, zero beyond 16 samples."""
    inside = np.abs(t) < W_HALF
    arg = beta * np.sqrt(np.where(inside, 1 - (t / W_HALF) ** 2, 0.0))
    return np.where(inside, np.i0(arg) / np.i0(beta), 0.0)


def kernel(t):
    """The yippy.sinc Kaiser-sinc kernel, from the library."""
    return np.asarray(weights(jnp.asarray(t, dtype=jnp.float64)))


def truncated(t):
    """The ideal sinc with every weight beyond 16 samples dropped."""
    return np.where(np.abs(t) < W_HALF, np.sinc(t), 0.0)


def tone_error(freq, rule, offsets=np.linspace(0.0, 1.0, 41)):
    """Worst error over query offsets, for a unit tone at each frequency."""
    taps = np.arange(-15, 17)
    w = rule(offsets[:, None] - taps)
    tone = np.exp(2j * np.pi * freq[:, None] * taps)
    truth = np.exp(2j * np.pi * freq[:, None] * offsets)
    return np.max(np.abs(tone @ w.T - truth), axis=1)
```

## What does the cut remove?

A read at a query adds the 32 samples nearest it, each times a weight taken
from the sinc. The figure draws those weights for a query halfway between two
samples. Gray dots are the 32 taps the stencil keeps, and the pink rings are
the first weights it throws away.

```{code-cell} python
:tags: [hide-input]
taps = np.arange(-25, 27)
offset = 0.5 - taps
kept = (taps >= -15) & (taps <= 16)
w_sinc = np.sinc(offset)
first_dropped = np.abs(w_sinc[~kept][np.argmin(np.abs(offset[~kept]))])
dropped_big = np.abs(w_sinc[~kept]).max()

fig, ax = plt.subplots(figsize=(9, 2.8), constrained_layout=True)
ax.axhline(0, color=FAINT, lw=1)
ax.vlines(offset[kept], 0, w_sinc[kept], color=KEPT, lw=1.4)
ax.plot(offset[kept], w_sinc[kept], "o", color=KEPT, ms=4, label="kept: 32 taps")
ax.vlines(offset[~kept], 0, w_sinc[~kept], color=pal.pink, lw=1.2)
ax.plot(offset[~kept], w_sinc[~kept], "o", mfc="none", mec=pal.pink, ms=5,
        label="dropped by the cut")
ax.axvline(16, color=FAINT, lw=1, ls="--")
ax.axvline(-16, color=FAINT, lw=1, ls="--")
ax.set_xlim(-26, 26)
ax.set_ylim(-0.12, 0.3)
ax.set_xlabel("offset of the tap from the query (samples)")
ax.set_ylabel(r"weight $\mathrm{sinc}(t)$")
ax.set_title(rf"the first dropped weight has size ${first_dropped:.3f}$", color=pal.pink,
             fontsize=10)
ax.legend(loc="upper right", frameon=False)
plt.show()
num("k_first_dropped", f"{first_dropped:.3f}")
num("k_first_dropped_pct", f"{100 * first_dropped:.0f}")
```

Look at the pink rings just outside the dashed edges. They are as large as
{glue:text}`k_first_dropped` and they alternate in sign, so for a slow signal
they nearly cancel and the cut leaves an error near
{glue:text}`k_first_dropped_pct` percent, the size measured on the foundations
page. To lose less, the weights must already be small where the cut falls.

A smooth function that is 1 at the centre and falls close to zero at the edge,
multiplied into the sinc, does that. Such a function is called a **window**.

## How small can a window make the weights at the edge?

The **Kaiser-sinc kernel** is the sinc times a Kaiser window. The window has one
knob, a shape parameter $\beta$. The figure shows the window for four values,
and then the kernel that the largest one builds, in the same stacked axes so
that every curve is read at the same offset.

```{code-cell} python
:tags: [hide-input]
t = np.linspace(-17.5, 17.5, 3501)
fig, axes = plt.subplots(3, 1, figsize=(9, 7.4), sharex=True, constrained_layout=True)

top, mid, bot = axes
for beta, color, lw in ((0, CUT, 1.2), (4, CUT, 1.2), (8, CUT, 1.2), (12, pal.yellow, 2.2)):
    top.semilogy(t, np.maximum(kaiser(t, beta), 1e-12), color=color, lw=lw)
    edge = kaiser(np.array([15.999]), beta)[0]
    top.text(17.8, max(edge, 3e-6), rf"$\beta={beta}$", color=color, fontsize=9, va="center")
top.set_ylim(1e-6, 2)
top.set_ylabel("window")
top.set_title(r"the window at each shape parameter $\beta$", fontsize=10, color=SCENERY)

mid.plot(t, np.sinc(t), color=CUT, lw=1.2)
mid.plot(t, kernel(t), color=pal.cyan, lw=1.8)
mid.set_ylabel("weight")
mid.set_title("the kernel (cyan) beside the ideal sinc (gray)", fontsize=10, color=SCENERY)

bot.semilogy(t, np.maximum(np.abs(np.sinc(t)), 1e-12), color=CUT, lw=1.2)
bot.semilogy(t, np.maximum(np.abs(kernel(t)), 1e-12), color=pal.cyan, lw=1.8)
bot.set_ylim(1e-9, 2)
bot.set_ylabel("size of the weight")
bot.set_xlabel("offset of the tap from the query (samples)")
bot.set_title("the same two curves, sizes on a logarithmic axis", fontsize=10, color=SCENERY)
for ax in axes:
    for edge in (-16, 16):
        ax.axvline(edge, color=FAINT, lw=1, ls="--")
    ax.set_xlim(-17.5, 19.5)
plt.show()

i0_beta = np.i0(BETA)
edge_value = 1 / i0_beta
estimate = first_dropped * edge_value
formula_gap = float(np.max(np.abs(kernel(t) - np.sinc(t) * kaiser(t))))
num("k_i0", f"{i0_beta:.2g}")
glue("k_edge", tex(edge_value, 1), display=False)
glue("k_estimate", tex(estimate, 1), display=False)
glue("k_gap", tex(formula_gap, 0), display=False)
```

The top axes show every window starting at 1, but the larger $\beta$ is, the
lower it falls by the edge, at $t=\pm16$. The middle axes show that the
kernel keeps the central lobes of the sinc almost unchanged. The bottom axes show
what the window buys: the sinc's weights stay near
{glue:text}`k_first_dropped` all the way to the edge, and the kernel's fall
by several orders of magnitude.

In symbols, the kernel multiplies the sinc by the window,

$$
w(t)=\operatorname{sinc}(t)\,\frac{I_0\!\left(\beta\sqrt{1-(t/W)^2}\right)}{I_0(\beta)}
\quad\text{for } |t|<W,
$$

and is zero otherwise. Here $W=16$ is the half-width and $I_0$ is the modified
Bessel function. The window is 1 at $t=0$ and falls to $1/I_0(\beta)$ at the
edge, which for $\beta=12$ is {glue}`k_edge`, because $I_0(12)$ is about
{glue:text}`k_i0`. The library's weights agree with this formula to
{glue}`k_gap` at every point of the figure.

```{admonition} Predict before reading on
:class: tip
The cut sinc left an error set by the edge weight, {glue:text}`k_first_dropped`.
The window multiplies that weight by {glue}`k_edge`. Is the error left near
$10^{-2}$, $10^{-4}$ or $10^{-6}$ of the signal's amplitude?
```

```{dropdown} Answer
Near $10^{-6}$: {glue:text}`k_first_dropped` times {glue}`k_edge` is
{glue}`k_estimate`. The next section measures the error and finds it within a
factor of three of this estimate.
```

## Where does the error stay small?

The measure that follows is the **worst-tone error**: a single cosine of size 1
and frequency $f$ is sampled at the integers and read back at a query, and the
worst difference over every query position inside one cell, the interval
between two samples, is the error. The figure plots it against $f$ for the
cut sinc (gray) and for the kernel (cyan). The top axis gives the same
frequency as the oversampling factor $\rho=2f$ of
{doc}`the sampling page <../foundations/sampling>`.

```{code-cell} python
:tags: [hide-input]
freq = np.linspace(0.0, 0.5, 501)
err_kernel = tone_error(freq, kernel)
err_cut = tone_error(freq, truncated)
rho_of = lambda f: 2 * f

# Estimate of the band edge from Kaiser's design rules, in the dropdown below.
atten = BETA / 0.1102 + 8.7
floor_est = 10 ** (-atten / 20)
transition = (atten - 8) / (2.285 * 31) / (2 * np.pi)
edge_est = 0.5 - transition / 2

running = np.maximum.accumulate(err_kernel)
in_margin = freq <= 0.38 + 1e-9
cross = freq[np.argmax((err_kernel > err_cut) & (freq > 0.2))]

fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
ax.axvspan(0, 0.38, color=pal.cyan, alpha=0.06, lw=0)
ax.semilogy(freq, err_cut, color=CUT, lw=1.8)
ax.semilogy(freq, err_kernel, color=pal.cyan, lw=2.2)
ax.axhline(estimate, color=pal.yellow, lw=1, ls=":")
ax.axvline(0.38, color=pal.pink, lw=1.2, ls="--")
ax.axvline(0.5, color=SCENERY, lw=1)
ax.set_ylim(1e-8, 3)
ax.set_xlim(0, 0.52)
ax.text(0.015, 5e-1, "cut sinc", color=CUT, fontsize=10)
ax.text(0.015, 2.5e-5, "Kaiser-sinc kernel", color=pal.cyan, fontsize=10)
ax.text(0.015, estimate * 1.6, "edge-weight estimate", color=pal.yellow, fontsize=9)
ax.text(0.385, 2e-8, r"$f=0.38$", color=pal.pink, fontsize=9)
ax.text(0.497, 2e-8, "Nyquist", color=SCENERY, fontsize=9, ha="right")
ax.set_xlabel(r"tone frequency $f$ (cycles per sample)")
ax.set_ylabel("worst-tone error")
top_axis = ax.secondary_xaxis("top", functions=(rho_of, lambda r: r / 2))
top_axis.set_xlabel(r"oversampling factor $\rho=2f$")
plt.show()

at_kernel = lambda f: err_kernel[np.argmin(np.abs(freq - f))]
glue("k_floor_lo", tex(err_kernel[0], 1), display=False)
glue("k_floor", tex(err_kernel[in_margin].max(), 1), display=False)
cut_running = np.maximum.accumulate(err_cut)
num("k_cut_015_pct", f"{100 * cut_running[np.argmin(np.abs(freq - 0.15))]:.0f}")
num("k_cut_038_pct", f"{100 * cut_running[np.argmin(np.abs(freq - 0.38))]:.0f}")
glue("k_at_040", tex(at_kernel(0.40), 1), display=False)
glue("k_at_045", tex(at_kernel(0.45), 1), display=False)
num("k_at_050", f"{at_kernel(0.5):.2f}")
num("k_cross_rho", f"{rho_of(cross):.2f}")
num("k_ratio", f"{err_kernel[np.argmin(np.abs(freq - 0.4))] / err_kernel[in_margin].max():.0f}")
glue("k_est_floor", tex(floor_est, 1), display=False)
num("k_atten", f"{atten:.0f}")
num("k_trans", f"{transition:.2f}")
num("k_edge_est", f"{edge_est:.2f}")
num("k_rho_est", f"{rho_of(edge_est):.2f}")
```

Look at the cyan curve against the dotted yellow line. The kernel's error
sits between {glue}`k_floor_lo` at $f=0$ and {glue}`k_floor` at $f=0.38$,
near the yellow estimate of the previous section. The gray ripple of the cut sinc
peaks near {glue:text}`k_cut_015_pct` percent by $f=0.15$ and
{glue:text}`k_cut_038_pct` percent by $f=0.38$. Past $f=0.38$ the kernel's error climbs: {glue}`k_at_040` at
$f=0.40$, {glue}`k_at_045` at $f=0.45$ and {glue:text}`k_at_050` at the
Nyquist limit, the whole amplitude of the tone. The foundations figure of the
bandwidth-0.5 row shows why that last value must be 1: a tone at the Nyquist
limit can be zero at every sample.

The taper has a price. Beyond $\rho=${glue:text}`k_cross_rho` the kernel's error
is larger than the cut sinc's, so the repair buys a very small error inside
the band and costs accuracy near the Nyquist limit.

```{dropdown} The estimate of where the floor ends
Kaiser's design rules for filters give two numbers. His first rule relates
the shape parameter to the rejection the window achieves, $\beta=0.1102(A-8.7)$,
so $\beta=12$ gives $A\approx{}$ {glue:text}`k_atten` dB, an amplitude ratio near
{glue}`k_est_floor`, close to the edge-weight estimate. His length rule,
$N-1=(A-8)/(2.285\,\Delta\omega)$, gives the width of the transition between
passing and rejecting frequencies for $N=32$ taps: {glue:text}`k_trans` cycles
per sample. The transition is centred on the Nyquist limit, so the flat part ends
half a transition below it, at about {glue:text}`k_edge_est` cycles per sample,
which is $\rho\approx{}$ {glue:text}`k_rho_est`.

Kaiser derived both rules empirically for finite impulse response filters
(Kaiser 1974, and Oppenheim and Schafer, *Discrete-Time Signal Processing*,
section 7.6). They apply because interpolating at a fractional offset is
filtering with a shifted kernel, and the in-band ripple of that filter is the
error plotted above. The rules are approximations, and the measured curve is the
evidence.
```

## What margin does the kernel set?

The error is flat enough up to $f=0.38$ and then climbs by a factor of
{glue:text}`k_ratio` between there and $f=0.40$. yippy.sinc therefore admits a
grid only when $\rho\le0.76$ on every axis, and calls 0.76 the **margin**. It
is a chosen value: it is where the measured curve leaves its floor, and it
matches the Kaiser estimate. Inside the margin the worst-tone error is at most
{glue}`k_floor`. A smaller margin would lose nothing in accuracy but would
demand a finer grid; a larger one would meet the climb. {doc}`../practice/admission`
shows the refusal for a grid outside it.

```{dropdown} Why the worst tone bounds every signal
A signal that is a sum of tones is read as the sum of the tones' reads, so its
error is at most the largest tone error times the sum of the tone sizes. The
worst-tone error is therefore the guarantee for any signal of that bandwidth.
A random sum of twelve cosines does better, because its tones do not all err in
the same direction: the worst of 40 such signals at $\rho=0.76$ errs by
{glue}`k_random_76`, and at $\rho=1$ by {glue:text}`k_random_100`.
```

```{code-cell} python
:tags: [remove-cell]
glue("k_random_76", tex(tc.kernel_error(0.76), 1), display=False)
num("k_random_100", f"{tc.kernel_error(1.0):.2f}")
```

The guarantee assumes a signal that is exactly band-limited, sampled on a
regular grid with no gaps, and read at a query whose whole stencil exists: at
least 15 samples above the first node and more than 15 below the last. If the
first assumption fails the error is no longer controlled. For the other two, the
library refuses the read. The error is also a fraction of the signal's total
amplitude, not of its value at the query. {doc}`band-limits` measures what that
means for a coronagraph pixel that varies over ten orders of magnitude.

## Try it

1. Write the window as a function of $\beta$ (the formula above) and use it
   as the `kernel=` argument of `tc.kernel_error_worst` for $\beta=8$. Predict
   first, using Kaiser's rule: how high does the floor rise, and does the cliff
   move left or right?
   ```{dropdown} Hint
   $A=8/0.1102+8.7\approx81$ dB, an amplitude near $9\times10^{-5}$. The
   transition narrows, so the flat part ends closer to the Nyquist limit and
   the cliff moves right.
   ```
2. Repeat the error curve with $W=8$, a 16-tap kernel (zero beyond 8). Use the
   length rule to predict where the cliff starts before running it.
   ```{dropdown} Hint
   Halving the taps doubles the transition width, to about 0.5 cycles per
   sample, so the flat response ends near $0.5-0.25=0.25$ cycles per sample:
   $\rho\approx0.5$.
   ```
3. Predict, then read the figure: at which tone frequency does the cut sinc
   stop being worse than the kernel?
   ```{dropdown} Hint
   Where the cyan curve crosses the gray one, at $\rho=$ {glue:text}`k_cross_rho`.
   ```

## What this page established

:::{admonition} What perfect interpolation needs
:class: important

1. **A band limit.** The signal has no cosine above a known bandwidth
   $f_{\max}$.
2. **Dense enough samples.** $\rho=2f_{\max}<1$.
3. **Weights a program can afford.** The sinc times a Kaiser window, kept to 32
   taps, holds the worst-tone error below {glue}`k_floor` while
   $\rho\le0.76$, and loses accuracy beyond it.
:::

The kernel is a smooth taper of the sinc, and its margin is $\rho=0.76$. The next
page ({doc}`toy-coronagraph`) introduces the small coronagraph used from here
on, whose planet images are the signals a grid has to pin down.

Next: {doc}`toy-coronagraph`.
