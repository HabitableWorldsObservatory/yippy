---
file_format: mystnb
kernelspec:
  name: python3
---

# How do samples give back the values between them?

The {doc}`previous page <sampling>` found that when a signal has no cosine
at or above 0.5 cycles per sample, its samples pin it down: exactly one such
curve passes through them. Knowing that the curve is determined is not the
same as knowing how to compute it at a position between samples. This page
builds that computation from scratch, and then asks what is lost when it has
to be done with a limited number of samples, as any real program must.

:::{admonition} What perfect interpolation needs
:class: important

1. **A band limit.** The signal has no cosine above a known bandwidth
   $f_{\max}$. *(previous page)*
2. **Dense enough samples.** $\rho=2f_{\max}<1$. *(previous page)*
3. **A way to combine the samples.** *This page.*
:::

```{code-cell} python
:tags: [remove-cell]
import tempfile
from pathlib import Path

import hwostyle
import jax
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Video
from matplotlib.collections import LineCollection

import eyepiece as ep
import toy_coronagraph as tc

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette
FAINT = ep.style.neutral(0.25)
SCENERY = ep.style.neutral(0.45)

_clips = iter(range(100))


def clip(anim):
    """Embed an animation as an mp4 with the browser's own player controls."""
    path = Path(tempfile.gettempdir()) / f"yippy-sinc-docs-sinc-{next(_clips)}.mp4"
    anim.video(path, dpi=200)
    return Video(str(path), embed=True, html_attributes="controls loop muted playsinline")

F_MAX = 0.38
rng = np.random.default_rng(3)
freqs = rng.uniform(-F_MAX, F_MAX, 12)
amps = rng.normal(size=12)
phases = rng.uniform(0, 2 * np.pi, 12)


def signal(t):
    """The test signal of the previous page: twelve cosines below 0.38."""
    t = np.asarray(t, dtype=float)[..., None]
    return np.sum(amps * np.cos(2 * np.pi * freqs * t + phases), axis=-1)


nodes = np.arange(-60, 61)
samples = signal(nodes)
```

## Each sample contributes one copy of a shape

A familiar method shows the idea. Linear interpolation, drawing straight
lines between samples, can be done in an unusual way: give every sample a
triangle that is 1 at the sample and falls to 0 at its two neighbors, scale
the triangle by the sample's value, and add all the triangles up.

```{code-cell} python
:tags: [hide-input]
n6 = np.arange(0, 7)
t6 = np.linspace(-1, 7, 1601)
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True,
                                  constrained_layout=True)
for n in n6:
    tri = signal(n) * np.clip(1 - np.abs(t6 - n), 0, None)
    left.plot(t6, tri, color=pal.yellow, lw=1.2, alpha=0.8)
left.plot(n6, signal(n6), "o", color=pal.cyan, ms=4.5)
left.set_title("one scaled triangle per sample", fontsize=11)
total = sum(signal(n) * np.clip(1 - np.abs(t6 - n), 0, None) for n in n6)
right.plot(t6, total, color=pal.yellow, lw=2)
right.plot(n6, signal(n6), "o", color=pal.cyan, ms=4.5)
right.set_title("their sum: straight lines", fontsize=11)
for ax in (left, right):
    ax.axhline(0, color=FAINT, lw=0.8)
    ax.set_xlim(-0.5, 6.5)
    ax.set_xlabel(r"position $t$ (samples)")
left.set_ylabel("value")
plt.show()
```

Every triangle is the same shape, moved to its own sample and scaled by that
sample's value. Writing the signal as $s(t)$ and the shape as $w$, the value
at any position $t$ is

$$
s(t)=\sum_{n} s(n)\,w(t-n),
$$

where $s(n)$ is the sample at the integer position $n$, and $t$, the position
being asked about, is called the **query**. Read aloud: each sample spreads
its value along the line in the pattern $w$, centered on itself, and the
value at the query is the total. Linear interpolation is this sum with $w$
a triangle. The question is which shape $w$ gives back the band-limited
curve instead of straight lines.

## What must the shape do?

The first test is about the samples themselves. The sum must reproduce every
sample exactly, so at a sample position only that sample may contribute. The
shape must therefore be 1 at zero and 0 at every other integer. The triangle
passes this test: at each sample, every other triangle has already fallen to
zero.

The second test is about frequency. Every copy of $w$ goes into the sum, so
any cosine that $w$ contains ends up in the result. The shape has to work for
every signal the grid can hold, which by the previous page means every signal
with nothing at or above the Nyquist limit, so $w$ must contain nothing above
0.5 cycles per sample. That limit is set by the grid, not by the 0.38 of this
particular signal. The triangle fails here, as its spectrum shows.

```{code-cell} python
:tags: [hide-input]
t_k = np.linspace(-6, 6, 1201)
f_k = np.linspace(0, 1.6, 801)
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 2.6), constrained_layout=True)
left.axhline(0, color=FAINT, lw=0.8)
left.plot(t_k, np.clip(1 - np.abs(t_k), 0, None), color=pal.yellow, lw=2)
ints = np.arange(-6, 7)
left.plot(ints, np.where(ints == 0, 1.0, 0.0), "o", color=pal.cyan, ms=4)
left.set_xlim(-6, 6)
left.set_xlabel(r"position $t$ (samples)")
left.set_ylabel(r"$w(t)$")
left.set_title("the triangle: passes the first test", fontsize=11)
spec = np.sinc(f_k) ** 2
right.plot(f_k, spec, color=pal.yellow, lw=1.6)
right.fill_between(f_k[f_k > 0.5], spec[f_k > 0.5], color=pal.yellow, alpha=0.4, lw=0)
right.axvline(0.5, color=SCENERY, lw=1, ls=":")
right.axhline(1.0, xmax=0.5 / 1.6, color=pal.pink, lw=1, ls="--")
right.text(0.02, 1.05, "ideal: 1 up to 0.5", color=pal.pink, fontsize=9)
right.text(0.52, 0.75, "Nyquist limit", color=SCENERY, fontsize=9)
right.text(0.75, 0.22, "cosines above 0.5", color=pal.yellow, fontsize=9)
right.set_xlim(0, 1.6)
right.set_ylim(0, 1.15)
right.set_xlabel("frequency (cycles per sample)")
right.set_ylabel("size")
right.set_title("its spectrum: fails the second", fontsize=11)
plt.show()
```

The right panel is the triangle's spectrum, the sizes of the cosines that
build it. A single pulse that never repeats needs a continuous range of
frequencies, so its spectrum is a curve rather than separate stems. Two
things are wrong with it. The shaded part lies above the Nyquist limit:
cosines that linear interpolation adds to every result, coming from the
triangle's sharp corners. And inside the band the curve droops below 1: a
cosine at 0.38 cycles per sample comes through at only 61 percent of its
size, with faster cosines making up the rest. A shape that interpolates
perfectly would need a spectrum of exactly 1 up to 0.5 and 0 beyond (dashed).

```{dropdown} Where the triangle's spectrum comes from
The triangle is the shape you get by sliding a box of width 1 across another
box of width 1 and recording their overlap. A box's spectrum is
$\sin(\pi f)/(\pi f)$, and sliding one shape across another multiplies
spectra, so the triangle's spectrum is $\big(\sin(\pi f)/(\pi f)\big)^2$
(Bracewell, *The Fourier Transform and Its Applications*, chapter 4).
```

## A shape built only from allowed cosines

The second test says the shape may only contain cosines up to 0.5 cycles per
sample, so build it from exactly those. The first test wants a peak of 1 at
zero, and every cosine with no phase shift starts at its peak at $t=0$, so
use cosines that all start there. With no reason to favor any frequency over
another, take them in equal amounts and average them. The animation averages
more and more of them, evenly spaced in frequency between 0 and 0.5. The
faint lines are some of the individual cosines and the pink curve is their
average.

```{code-cell} python
:tags: [hide-input]
# An animation: the average's zero crossings settle onto the integers while
# its peak stays at 1. The still follows.
t_s = np.linspace(-6, 6, 1001)
ints = np.arange(-6, 7)


def cosines(count):
    """Cosines at frequencies evenly spread over 0 to 0.5 cycles per sample."""
    f = (np.arange(count) + 0.5) / count * 0.5
    return np.cos(2 * np.pi * f[:, None] * t_s[None, :])


def averaging_axes():
    fig, ax = plt.subplots(figsize=(7.5, 2.8), constrained_layout=True)
    ax.axhline(0, color=FAINT, lw=0.8)
    ax.plot(ints, np.where(ints == 0, 1.0, 0.0), "o", color=pal.cyan, ms=4, zorder=5)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1.05, 1.15)
    ax.set_xlabel(r"position $t$ (samples)")
    ax.set_ylabel("value")
    return fig, ax


fig, ax = averaging_axes()
members = LineCollection([], colors=FAINT, linewidths=0.8)
ax.add_collection(members)
(avg,) = ax.plot([], [], color=pal.pink, lw=2.2)
readout = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=10)


def draw(fig, count):
    rows = cosines(count)
    members.set_segments([np.column_stack([t_s, r]) for r in rows[:: max(1, count // 8)]])
    avg.set_data(t_s, rows.mean(axis=0))
    readout.set_text(f"cosines averaged: {count}")


counts = [1, 2, 3, 4, 6, 8, 11, 16, 24, 40, 80]
frames = [1] + counts + [80] * 3
anim = ep.render.animate(fig, draw, frames, fps=1)
clip(anim)
```

```{code-cell} python
:tags: [hide-input]
fig, ax = averaging_axes()
for row in cosines(80)[::10]:
    ax.plot(t_s, row, color=FAINT, lw=0.8)
ax.plot(t_s, cosines(80).mean(axis=0), color=pal.pink, lw=2.2)
ax.text(0.01, 0.97, "cosines averaged: 80", transform=ax.transAxes, va="top", fontsize=10)
plt.show()
```

As more cosines join, the average settles into a fixed shape with a peak of 1
at zero and zero crossings that land on the integers (cyan dots). To see why
the crossings land there, freeze the position and look across the cosines
instead. Each panel below fixes one position $t$ and plots the value of every
allowed cosine there, against that cosine's frequency. The average at that
position is the shaded area above zero (cyan) minus the area below (pink),
divided by the width of the band.

```{code-cell} python
:tags: [hide-input]
f_band = np.linspace(0, 0.5, 400)
fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), sharey=True, constrained_layout=True)
for ax, t0 in zip(axes, (0.0, 1.0, 2.0, 1.5)):
    vals = np.cos(2 * np.pi * f_band * t0)
    ax.axhline(0, color=FAINT, lw=0.8)
    ax.fill_between(f_band, vals, where=vals >= 0, color=pal.cyan, alpha=0.4, lw=0)
    ax.fill_between(f_band, vals, where=vals < 0, color=pal.pink, alpha=0.4, lw=0)
    ax.plot(f_band, vals, color=SCENERY, lw=1.2)
    ax.set_title(rf"at $t={t0:g}$: average {vals.mean():.2f}", fontsize=11)
    ax.set_xlabel("cosine frequency")
    ax.set_xlim(0, 0.5)
axes[0].set_ylabel("cosine value")
plt.show()
```

At $t=0$ every cosine is at its peak, so the average is 1. At $t=1$ the
cosines have turned by anything from no turn to half a turn. The second half
of that range mirrors the first with the sign flipped, because
$\cos(\pi-x)=-\cos x$, so the pink area exactly cancels the cyan. At $t=2$
the turns run from none to a full turn, which is two such mirrored halves,
and at any whole number $t$ the range is a whole number of half turns that
cancel in pairs. At $t=1.5$ the range is three quarters of a turn, which
does not split into mirrored halves, so the average is not zero.

The averaged shape is the **sinc**,

$$
\operatorname{sinc}(t)=\frac{\sin \pi t}{\pi t},
$$

which is what the average of all cosines from 0 to 0.5 works out to exactly.
It passes the first test, as the figure showed, and it passes the second by
construction: it contains every cosine up to 0.5 in equal amounts, a spectrum
of exactly 1 up to the Nyquist limit and nothing above. In fact no other
shape passes both tests.

```{dropdown} The average, worked out
Averaging over the band of width $1/2$ means integrating and multiplying by
2:

$$
2\int_0^{1/2}\cos(2\pi f t)\,df
=2\left[\frac{\sin(2\pi f t)}{2\pi t}\right]_0^{1/2}
=\frac{\sin(\pi t)}{\pi t}.
$$

At a whole number $t=n\neq0$ the numerator is $\sin(\pi n)=0$, which is the
cancellation of the figure.
```

```{dropdown} Why no other shape passes both tests
Sampling a shape at the integers adds its spectrum to copies of itself
shifted by 1, 2, 3, ... cycles per sample (the same aliasing as the previous
page). The first test says the samples of $w$ are a single 1 at zero, whose
spectrum is flat at 1, so the shifted copies must add up to exactly 1 at
every frequency. The second test says $w$ has nothing outside $-0.5$ to
$0.5$, so the copies do not overlap, and each must be exactly 1 inside its
own band. That is the sinc's spectrum, so the shape is the sinc.
```

```{code-cell} python
:tags: [hide-input]
t_k = np.linspace(-8, 8, 2001)
f_k = np.linspace(0, 1.6, 801)
tri = np.clip(1 - np.abs(t_k), 0, None)
fig, axes = plt.subplots(2, 2, figsize=(9, 4.6), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.5, 1]})
rows = (
    ("sinc", np.sinc(t_k), (f_k <= 0.5).astype(float), pal.pink),
    ("triangle", tri, np.sinc(f_k) ** 2, pal.yellow),
)
for row, (name, shape, spec, color) in enumerate(rows):
    ax, sx = axes[row]
    ax.axhline(0, color=FAINT, lw=0.8)
    ax.plot(t_k, shape, color=color, lw=2)
    ints = np.arange(-8, 9)
    ax.plot(ints, np.where(ints == 0, 1.0, 0.0), "o", color=pal.cyan, ms=4)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.3, 1.12)
    ax.set_ylabel(r"$w(t)$")
    ax.text(7.8, 0.85, name, color=color, ha="right")
    sx.fill_between(f_k, spec, color=color, alpha=0.35, lw=0)
    sx.plot(f_k, spec, color=color, lw=1.6)
    sx.axvline(0.5, color=SCENERY, lw=1, ls=":")
    sx.set_xlim(0, 1.6)
    sx.set_ylim(0, 1.15)
    sx.set_ylabel("size")
for side in (t_k > 1.2, t_k < -1.2):
    axes[0, 0].plot(t_k[side], 1 / (np.pi * np.abs(t_k[side])), color=SCENERY, lw=1,
                    ls="--")
axes[0, 0].text(3.0, 0.2, r"$1/(\pi |t|)$", color=SCENERY, fontsize=10)
axes[1, 0].set_xlabel(r"position $t$ (samples)")
axes[1, 1].set_xlabel("frequency (cycles per sample)")
axes[0, 1].set_title("spectrum", fontsize=11)
axes[0, 1].text(0.53, 1.05, "Nyquist limit", color=SCENERY, fontsize=9, va="top")
plt.show()
```

Side by side, the sinc (top) and the triangle (bottom) both pass through 1 at
zero and 0 at the other integers, but only the sinc's spectrum is flat to the
Nyquist limit and zero beyond. The price is in the tails. The triangle is
zero beyond one sample, while the sinc keeps rippling on both sides, and its
ripples shrink only as $1/(\pi|t|)$, the dashed envelope: ten samples away a
ripple is still about 3 percent of the peak.

## How many samples does one value need?

```{admonition} Predict before reading on
:class: tip
Use the sinc as the shape $w$. To compute the value at $t=0.3$, which
samples contribute: only the two nearest, the nearest ten, or every sample?
```

```{dropdown} Answer
Every sample. The sample at $n=10$ sits $9.7$ units from the query, and its
copy of the sinc has a ripple of size $|\operatorname{sinc}(9.7)|=0.027$
there, close to the envelope value $1/(9.7\pi)=0.033$. Halving the distance
only doubles the contribution, so distant samples matter a little, and there
are many of them.
```

The animation rebuilds the test signal from sinc copies, adding the samples
nearest $t=0.3$ first. The top panel shows the copy added at each step
(yellow): the sinc moved to one sample and scaled by that sample's value.
The panel below shows the sum of every copy added so far (pink) over the
true signal (gray), on the same scale. Watch the pink curve lock onto each
sample as soon as that sample's copy arrives, while the stretches between
samples keep shifting as more distant copies arrive.

```{code-cell} python
:tags: [hide-input]
# An animation: the running sum pins to each new sample while the values
# between samples keep drifting. The slow settling is the point, and no single
# still carries a rate.
QUERY = 0.3
by_distance = nodes[np.argsort(np.abs(nodes - QUERY), kind="stable")]
t_win = np.linspace(-7, 8, 1000)
truth = signal(t_win)
copies = np.sinc(t_win[None, :] - by_distance[:, None]) * signal(by_distance)[:, None]
partial = np.cumsum(copies, axis=0)
at_query = np.cumsum(np.sinc(QUERY - by_distance) * signal(by_distance))
win = (nodes >= -7) & (nodes <= 8)


def rebuild_axes():
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.5, 4.6), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1.3]},
    )
    for ax in (top, bottom):
        ax.axhline(0, color=FAINT, lw=0.8)
        ax.axvline(QUERY, color=SCENERY, lw=0.8, ls=":")
        ax.set_ylim(truth.min() - 0.6, truth.max() + 1.2)
        ax.set_ylabel("value")
    bottom.plot(t_win, truth, color=SCENERY, lw=1.2)
    bottom.set_xlim(-7, 8)
    bottom.set_xlabel(r"position $t$ (samples)")
    bottom.set_title("running sum", color=pal.pink, fontsize=11)
    bottom.text(7.8, truth.max() + 0.3, "true signal", color=SCENERY, ha="right", fontsize=9)
    return fig, top, bottom


fig, top, bottom = rebuild_axes()
(newest,) = top.plot([], [], color=pal.yellow, lw=1.6)
(newest_dot,) = top.plot([], [], "o", color=pal.yellow, ms=6)
new_title = top.set_title("", color=pal.yellow, fontsize=11)
bottom.plot(nodes[win], samples[win], "o", color=FAINT, ms=4)
(running,) = bottom.plot([], [], color=pal.pink, lw=2)
(used_dots,) = bottom.plot([], [], "o", color=pal.cyan, ms=4.5)
readout = bottom.text(0.01, 0.97, "", transform=bottom.transAxes, va="top", fontsize=10)


def draw(fig, k):
    n = by_distance[k - 1]
    newest.set_data(t_win, copies[k - 1])
    newest_dot.set_data([n], [signal(n)] if -7 <= n <= 8 else [np.nan])
    new_title.set_text(f"copy {k} being added: the sinc moved to sample {n} and scaled by its value")
    running.set_data(t_win, partial[k - 1])
    used = np.sort(by_distance[:k])
    used = used[(used >= -7) & (used <= 8)]
    used_dots.set_data(used, signal(used))
    err = abs(at_query[k - 1] - signal(QUERY))
    readout.set_text(f"copies added: {k}    error at t = 0.3: {err:.3f}")


counts = [1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 28, 36, 50, 70, 95, 121]
frames = [1] + counts + [121] * 2
anim = ep.render.animate(fig, draw, frames, fps=3)
clip(anim)
```

```{code-cell} python
:tags: [hide-input]
fig, top, bottom = rebuild_axes()
for n in nodes[win]:
    top.plot(t_win, np.sinc(t_win - n) * signal(n), color=pal.cyan, lw=0.9, alpha=0.7)
top.set_title("the copies from the samples in view", color=pal.cyan, fontsize=11)
bottom.plot(t_win, partial[-1], color=pal.pink, lw=1.5)
bottom.plot(nodes[win], samples[win], "o", color=pal.cyan, ms=4.5)
bottom.set_title(f"sum of all {len(nodes)} copies", color=pal.pink, fontsize=11)
bottom.text(0.01, 0.97, f"copies added: {len(nodes)}", transform=bottom.transAxes, va="top",
            fontsize=10)
plt.show()
```

With all 121 copies added, the sum (pink) lies on the true signal (gray)
across the window. The error at the query does not reach zero: it settles
near 0.008, about half a percent of the signal's typical size, because the
samples beyond the 121 used are still missing. The first missing sample is
about 60 units away, where the sinc's envelope is $1/(60\pi)\approx0.005$,
so the leftover is the size one expects.

Carried out over every sample, the procedure is the Whittaker-Shannon
interpolation formula,

$$
s(t)=\sum_{n=-\infty}^{\infty} s(n)\,\operatorname{sinc}(t-n),
$$

which reads: the value at $t$ is every sample multiplied by the sinc of its
distance from $t$, all added up. The terms shrink only as one over the
distance, so the sum converges slowly, which is the slow settling of the
animation. The multiplier $\operatorname{sinc}(t-n)$ is called the
**weight** of sample $n$ for the query $t$. This is the third requirement:
**use every sample, weighted by the sinc of its distance**. With all three
met the reconstruction is exact, which is the content of the sampling
theorem (Goodman, *Introduction to Fourier Optics*, section 2.4).

## A real program can use only nearby samples

The formula needs every sample out to infinity, and a real grid is finite.
Speed matters too: a planet image is a two-dimensional function, so a
two-dimensional value multiplies the count, and 32 samples along each axis
already means 1024 samples per value. The code that interpolates (yippy.sinc
calls it the reader) therefore uses only the samples nearest the query. Each
sample it uses is a **tap**, and the set of taps is the **stencil**.
yippy.sinc's stencil holds 32 taps, 16 on each side of the query, a balance
between cost and accuracy that {doc}`../theory/kernel` measures.

```{code-cell} python
:tags: [hide-input]
def stencil_of(q, half=16):
    """The 32 integer taps nearest a query: 16 at or below it, 16 above."""
    return np.floor(q) + np.arange(-half + 1, half + 1)


taps = stencil_of(QUERY)
around = np.arange(-24, 26)
fig, ax = plt.subplots(figsize=(8.5, 2.8), constrained_layout=True)
ax.axvspan(taps[0] - 0.5, taps[-1] + 0.5, color=pal.cyan, alpha=0.08, lw=0)
ax.axhline(0, color=FAINT, lw=0.8)
outside = around[(around < taps[0]) | (around > taps[-1])]
ax.vlines(outside, 0, np.sinc(QUERY - outside), color=FAINT, lw=1.6)
ax.plot(outside, np.sinc(QUERY - outside), "o", mfc="none", mec=FAINT, ms=4)
ax.vlines(taps, 0, np.sinc(QUERY - taps), color=pal.cyan, lw=1.8)
ax.plot(taps, np.sinc(QUERY - taps), "o", color=pal.cyan, ms=4)
ax.axvline(QUERY, color=pal.pink, lw=1.4)
ax.text(QUERY + 0.4, 1.02, r"query $t=0.3$", color=pal.pink, va="top")
ax.text(taps[-1] - 0.3, 0.55, "stencil: 32 taps", color=pal.cyan, ha="right")
w3 = np.sinc(QUERY - 3)
ax.annotate(rf"tap $n=3$: weight $\mathrm{{sinc}}(0.3-3)={w3:.3f}$",
            xy=(3, w3), xytext=(4.2, -0.32), color=SCENERY, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=SCENERY, lw=0.8))
ax.text(-23.5, 0.3, "dropped", color=SCENERY, fontsize=9)
ax.text(24.5, 0.3, "dropped", color=SCENERY, fontsize=9, ha="right")
ax.set_xlim(-24.5, 25.5)
ax.set_ylim(-0.42, 1.08)
ax.set_xlabel(r"sample position $n$")
ax.set_ylabel("weight")
plt.show()
```

Each stem is one sample's weight for the query at $t=0.3$ (pink line). The 32
cyan stems inside the shaded stencil are the taps, and the value is the sum
of each tap's sample times its weight. The weights alternate in sign, so some
samples count negatively. The hollow gray stems outside are the weights a
32-tap reconstruction drops.

As the query moves, the weights change and the stencil moves with it. The
stencil is always the 16 samples at or below the query and the 16 above, so
when the query passes a sample, that sample changes sides and the stencil
slides by one. In the animation the query moves from sample 0 to sample 3.
Watch the moments it lands on a sample: every weight but one falls to zero,
and the reconstruction passes through that sample exactly.

```{code-cell} python
:tags: [hide-input]
# An animation: as the query crosses a sample, the weights collapse to a
# single 1 and the stencil jumps by one tap, which a strip of stills cannot
# follow across the jump.
def truncated_value(q):
    tp = stencil_of(q)
    return float(np.sum(np.sinc(q - tp) * signal(tp)))


t_drawn = np.linspace(0.0, 3.0, 301)
v_drawn = np.array([truncated_value(q) for q in t_drawn])
t_top = np.linspace(-4, 7, 500)


def sweep_axes():
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.5, 4.4), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.1, 1]},
    )
    top.plot(t_top, signal(t_top), color=SCENERY, lw=1.2)
    top.plot(nodes, samples, "o", color=pal.cyan, ms=3.5)
    top.set_xlim(-4, 7)
    top.set_ylabel("value")
    top.set_xlabel(r"position $t$ (samples), zoomed in")
    bottom.axhline(0, color=FAINT, lw=0.8)
    bottom.set_xlim(-20, 23)
    bottom.set_ylim(-0.3, 1.1)
    bottom.set_xlabel(r"sample position $n$")
    bottom.set_ylabel("weight")
    return fig, top, bottom


def draw_query(top, bottom, q, artists=None):
    tp = stencil_of(q)
    w = np.sinc(q - tp)
    if artists is None:
        artists = {
            "band": bottom.axvspan(-15.5, 16.5, color=pal.cyan, alpha=0.08, lw=0),
            "stems": bottom.add_collection(LineCollection([], colors=pal.cyan, linewidths=1.6)),
            "tips": bottom.plot([], [], "o", color=pal.cyan, ms=3)[0],
            "trace": top.plot([], [], color=pal.pink, lw=2.2)[0],
            "head": top.plot([], [], "o", color=pal.pink, ms=7)[0],
            "lines": (top.axvline(0.0, color=pal.pink, lw=1),
                      bottom.axvline(0.0, color=pal.pink, lw=1)),
            "readout": bottom.text(0.99, 0.95, "", transform=bottom.transAxes, ha="right",
                                   va="top"),
        }
    artists["stems"].set_segments([[(x, 0), (x, y)] for x, y in zip(tp, w)])
    artists["tips"].set_data(tp, w)
    artists["band"].set_x(tp[0] - 0.5)
    shown = t_drawn <= q
    artists["trace"].set_data(t_drawn[shown], v_drawn[shown])
    artists["head"].set_data([q], [truncated_value(q)])
    for line in artists["lines"]:
        line.set_xdata([q, q])
    artists["readout"].set_text(rf"query $t$ = {q:.2f}")
    return artists


fig, top, bottom = sweep_axes()
state = draw_query(top, bottom, 0.0)
queries = np.concatenate([np.zeros(1), np.linspace(0.0, 3.0, 25), np.full(2, 3.0)])
anim = ep.render.animate(fig, lambda fig, q: draw_query(top, bottom, q, state), queries,
                         fps=6)
clip(anim)
```

```{code-cell} python
:tags: [hide-input]
fig, top, bottom = sweep_axes()
draw_query(top, bottom, 2.2)
plt.show()
```

The still freezes the sweep at $t=2.2$: the trace has passed samples 1 and 2
exactly, and the stencil has slid two samples to the right of where it began.

## What does cutting the sinc at the stencil's edge cost?

```{admonition} Predict before reading on
:class: tip
Keep the 32 taps and set every weight beyond them to zero. For a single
cosine of size 1, at the worst frequency up to the bandwidth and the worst
query position, is the error near $10^{-2}$, $10^{-4}$ or $10^{-7}$?
```

The cell measures exactly that worst case for four values of $\rho$, using a
helper that sweeps every frequency up to the bandwidth and every query
position between two samples.

```{code-cell} python
:tags: [hide-input]
def truncated(t):
    """The ideal sinc with every weight beyond 16 samples set to zero."""
    return np.where(np.abs(t) < 16, np.sinc(t), 0.0)


for rho in (0.3, 0.5, 0.76, 0.9):
    worst = tc.kernel_error_worst(rho, kernel=truncated)
    print(f"rho = {rho}: worst single-cosine error {worst:.1e} of its size")
```

```{dropdown} Answer
Near $10^{-2}$: 2 percent at $\rho=0.3$, 5 percent at $\rho=0.76$ and nearly
10 percent at $\rho=0.9$. An estimate gives the size at low $\rho$. The first
dropped weight is 16 units away, where the sinc's envelope is
$1/(16\pi)\approx0.02$. The dropped weights alternate in sign, so for a slow
signal they mostly cancel and the first one sets the size, about 2 percent.
```

The error comes from the cut, not from the signal. For slow signals it stays
near the 2 percent set by the first dropped weight. Near the Nyquist limit
it grows, because a cosine that flips sign almost every sample flips in step
with the alternating weights, and the dropped tail then adds up instead of
cancelling.

The cut is also a corner in the weights, the same flaw as the triangle's,
and it lets the reconstruction leak past the band. The cure is
to bend the sinc smoothly down to zero at the edges instead of cutting it
off, which keeps the stencil at 32 taps while giving up a small, controlled
amount of accuracy. {doc}`../theory/kernel` measures how small, and how
close to the Nyquist limit it lets a grid go.

Everything here assumed noiseless samples on an evenly spaced grid of a
signal that is exactly band-limited. Energy above the bandwidth would alias
into the samples, and an uneven grid would move the sinc's zero crossings
off the samples.

## Try it

yippy.sinc's reader does exactly this on every query, with a tapered sinc in
place of the cut one and in two dimensions, so the exercises are the same
calculations it makes.

1. At $t=0.3$, what is the weight of the sample at $n=-20$? Estimate it from
   the $1/(\pi|t|)$ envelope before using the formula.
   ```{dropdown} Hint
   The distance is $20.3$. The envelope gives the size; the sine factor
   $\sin(\pi(0.3+20))$ gives the exact size and the sign.
   ```
   ```{dropdown} Answer
   The envelope gives $1/(20.3\pi)=0.016$. The exact weight is
   $\operatorname{sinc}(20.3)=\sin(20.3\pi)/(20.3\pi)=-0.013$.
   ```
2. A constant signal is a cosine at frequency 0, well inside the band, so the
   formula must reproduce it exactly. What must the sinc weights for any
   query add up to?
   ```{dropdown} Hint
   The samples of a constant $c$ are all $c$, so the formula gives $c$ times
   the sum of the weights.
   ```
   ```{dropdown} Answer
   1, for every query. The 121 weights used in the rebuild already add up to
   $0.99998$.
   ```
3. Predict, then change `QUERY = 0.3` in the rebuild cell to `QUERY = 0.5`,
   halfway between two samples. Does the error at the query settle faster or
   slower than before?
   ```{dropdown} Hint
   Every weight contains the factor $\sin(\pi(t-n))$. How large is it at
   $t=0.5$?
   ```
   ```{dropdown} Answer
   Slower. At $t=0.5$ that factor has size 1 for every sample, its largest,
   so the distant weights are as large as they can be.
   ```

## What this page established

:::{admonition} What perfect interpolation needs
:class: important

1. **A band limit.** The signal has no cosine above a known bandwidth
   $f_{\max}$.
2. **Dense enough samples.** $\rho=2f_{\max}<1$.
3. **Every sample, weighted by the sinc of its distance.** The value at $t$
   is $\sum_n s(n)\operatorname{sinc}(t-n)$ over all samples.
:::

The first two requirements are properties of the signal and the grid, and
{doc}`../theory/band-limits` shows that a coronagraph grid can meet them.
The third cannot be met by any finite grid or any practical program. A
32-tap stencil with the sinc cut off at its edges leaves errors of a few
percent, and the next page replaces the cut with a smooth taper and measures
how much accuracy that recovers.

Next: {doc}`../theory/kernel`.
