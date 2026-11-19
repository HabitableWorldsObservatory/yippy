---
file_format: mystnb
kernelspec:
  name: python3
---

# When do samples pin down a signal?

yippy.sinc computes coronagraph images for planet positions that lie between
the positions where images were precomputed. Seen from one detector pixel,
that is interpolation: the pixel's brightness is a function of the planet's
position, the precomputed images are samples of that function on a grid, and
every value a user asks for lies between samples. Everything yippy.sinc reports
therefore rests on one question: what has to be true, and what has to be
done, for the values between samples to come out exactly right?

:::{admonition} The question of these two pages
:class: important

What does perfect interpolation need? This page finds the first two
requirements, which are about the signal and the samples. The
{doc}`next page <sinc-interpolation>` finds the third, which is about how the
samples are combined, and then asks what is lost when it cannot be met in
full.
:::

```{code-cell} python
:tags: [remove-cell]
import tempfile
from pathlib import Path

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Video

import eyepiece as ep

hwostyle.use("dark")
pal = hwostyle.palette
FAINT = ep.style.neutral(0.25)
SCENERY = ep.style.neutral(0.45)

_clips = iter(range(100))


def clip(anim):
    """Embed an animation as an mp4 with the browser's own player controls."""
    path = Path(tempfile.gettempdir()) / f"yippy-sinc-docs-sampling-{next(_clips)}.mp4"
    anim.video(path, dpi=200)
    return Video(str(path), embed=True, html_attributes="controls loop muted playsinline")

F_MAX = 0.38
rng = np.random.default_rng(3)
freqs = rng.uniform(-F_MAX, F_MAX, 12)
amps = rng.normal(size=12)
phases = rng.uniform(0, 2 * np.pi, 12)


def component(k, t):
    """One cosine of the test signal."""
    return amps[k] * np.cos(2 * np.pi * freqs[k] * np.asarray(t, dtype=float) + phases[k])


def signal(t):
    """The test signal: the sum of its twelve cosines."""
    return sum(component(k, t) for k in range(12))
```

## Can the samples alone decide what lies between them?

Start with twelve samples of a smooth signal, taken one unit apart. The
figure shows three different curves that pass through exactly the same
twelve points: a slow, smooth curve, straight lines from point to point, and
a curve that swings up and down once between every pair of samples.

```{code-cell} python
:tags: [hide-input]
n12 = np.arange(0, 12)
t_fine = np.linspace(0, 11, 1200)
curves = {
    "a slow curve": signal(t_fine),
    "straight lines": np.interp(t_fine, n12, signal(n12)),
    "a fast curve": signal(t_fine) + 1.2 * np.sin(2 * np.pi * t_fine),
}
fig, axes = plt.subplots(1, 3, figsize=(9, 2.4), sharey=True, constrained_layout=True)
for ax, (name, curve), color in zip(axes, curves.items(), (pal.pink, pal.yellow, pal.green)):
    ax.plot(t_fine, curve, color=color, lw=1.6)
    ax.plot(n12, signal(n12), "o", color=pal.cyan, ms=4.5)
    ax.set_title(name, color=color)
    ax.set_xlabel(r"position $t$ (samples)")
axes[0].set_ylabel("value")
plt.show()
```

No. All three curves fit the twelve samples perfectly, and so do infinitely
many others, so the samples cannot tell us which curve is the real one. The
fast curve shows how easy such curves are to make: it is the slow curve plus
one extra wave that is zero at every sample, and this page comes back to it
once it has the tools to rule it out. To
pick one we need to know something about the signal itself. What separates
the three curves is how rapidly each one oscillates: the slow curve rises
and falls over several samples, while the fast curve rises and falls between
every pair of samples. The rest of this page makes "how rapidly a signal can
oscillate" precise and shows that knowing it, together with close enough
samples, picks out a single curve.

## A cosine repeats at one steady rate

The simplest signal that oscillates at one definite rate is a cosine. Its
**frequency** counts how many full cycles it completes per unit of position.
Positions on these pages are measured in units of the sample spacing, so
frequency comes out in cycles per sample.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 3, figsize=(9, 2.5), sharey=True, constrained_layout=True)
t_c = np.linspace(0, 12, 1200)
n_c = np.arange(0, 13)
for ax, f in zip(axes, (0.1, 0.25, 0.4)):
    ax.plot(t_c, np.cos(2 * np.pi * f * t_c), color=pal.cyan, lw=1.6)
    ax.plot(n_c, np.cos(2 * np.pi * f * n_c), "o", color=pal.cyan, ms=3.5)
    ax.set_title(f"{f} cycles per sample", fontsize=11)
    ax.set_xlabel(r"position $t$ (samples)")
period = 1 / 0.25
axes[1].annotate("", xy=(period, 1.25), xytext=(0, 1.25),
                 arrowprops=dict(arrowstyle="<->", color=SCENERY, lw=1))
axes[1].text(period + 0.3, 1.25, "one cycle = 4 samples", color=SCENERY, va="center",
             fontsize=9)
axes[0].set_ylabel("value")
axes[0].set_ylim(-1.2, 1.5)
plt.show()
```

The middle cosine completes one cycle every 4 samples, so its frequency is
$1/4=0.25$ cycles per sample. The left one needs 10 samples per cycle (0.1
cycles per sample), the right one only 2.5 (0.4 cycles per sample). A
frequency and its number of samples per cycle are reciprocals.

"Fast" on these pages means a high frequency: many turns per sample. It does
not mean steep. A tall cosine at 0.1 cycles per sample can have a steeper
slope than a small one at 0.4, but its shape still only changes over a
distance of several samples, and that distance, not the slope, is what
decides whether samples can capture it.

## Adding cosines builds other shapes

A cosine alone is a very particular shape, but adding several cosines with
different frequencies, sizes and starting points (**phases**) builds new
shapes. The animation builds the slow curve of the first figure this way,
one cosine at a time, starting from the slowest. The top panel shows the
cosine being added at each step (yellow), and the panel below shows the sum
of every cosine added so far (pink), with the target curve faint behind it.
Both use the same vertical scale, so the sizes compare directly. On the
right, each cosine appears as a stem whose position is its frequency and
whose height is its size: the step's cosine in yellow, earlier ones in cyan.

```{code-cell} python
:tags: [hide-input]
# An animation rather than a strip of stills: the reader follows the running
# sum taking shape as each cosine lights its stem, and twelve paired states are
# hard to match across a strip.
by_freq = np.argsort(np.abs(freqs))
t_b = np.linspace(0, 11, 700)
partials = np.cumsum([component(k, t_b) for k in by_freq], axis=0)
Y_LIM = (signal(t_b).min() - 0.8, signal(t_b).max() + 1.0)


def buildup_axes():
    fig, axd = plt.subplot_mosaic(
        [["new", "spec"], ["sum", "spec"]], figsize=(9, 3.9), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.5, 1]},
    )
    for name in ("new", "sum"):
        ax = axd[name]
        ax.axhline(0, color=FAINT, lw=0.8)
        ax.set_xlim(0, 11)
        ax.set_ylim(*Y_LIM)
        ax.set_ylabel("value")
    axd["new"].tick_params(labelbottom=False)
    axd["sum"].plot(t_b, signal(t_b), color=FAINT, lw=3)
    axd["sum"].set_xlabel(r"position $t$ (samples)")
    axd["sum"].set_title("running sum", color=pal.pink, fontsize=11)
    spec = axd["spec"]
    spec.vlines(np.abs(freqs), 0, np.abs(amps), color=FAINT, lw=2)
    spec.set_xlim(0, 0.5)
    spec.set_ylim(0, 2.3)
    spec.set_xlabel("frequency (cycles per sample)")
    spec.set_ylabel("size")
    spec.set_title("the recipe (sizes only)", fontsize=11)
    return fig, axd


fig, axd = buildup_axes()
(newest,) = axd["new"].plot([], [], color=pal.yellow, lw=1.8)
new_title = axd["new"].set_title("", color=pal.yellow, fontsize=11)
(running,) = axd["sum"].plot([], [], color=pal.pink, lw=2)
earlier = axd["spec"].vlines([], [], [], color=pal.cyan, lw=2.4)
current = axd["spec"].vlines([], [], [], color=pal.yellow, lw=3)


def stem(j):
    return [(abs(freqs[j]), 0), (abs(freqs[j]), abs(amps[j]))]


def draw(fig, k):
    j = by_freq[k - 1]
    newest.set_data(t_b, component(j, t_b))
    new_title.set_text(f"cosine {k} of 12 being added: {abs(freqs[j]):.2f} cycles per sample")
    running.set_data(t_b, partials[k - 1])
    earlier.set_segments([stem(i) for i in by_freq[: k - 1]])
    current.set_segments([stem(j)])


frames = [1, 1] + [k for k in range(1, 13) for _ in range(2)] + [12] * 3
anim = ep.render.animate(fig, draw, frames, fps=2)
clip(anim)
```

```{code-cell} python
:tags: [hide-input]
fig, axd = buildup_axes()
for k in range(12):
    axd["new"].plot(t_b, component(k, t_b), color=pal.cyan, lw=1, alpha=0.7)
axd["new"].set_title("all twelve cosines", color=pal.cyan, fontsize=11)
axd["sum"].plot(t_b, partials[-1], color=pal.pink, lw=2)
axd["sum"].set_title("their sum", color=pal.pink, fontsize=11)
axd["spec"].vlines(np.abs(freqs), 0, np.abs(amps), color=pal.cyan, lw=2.4)
plt.show()
```

With all twelve cosines in, the pink sum lies on the slow curve. The still
shows the twelve cosines together on top, in the same cyan as their stems,
and their sum below. The list of frequencies and sizes is the curve's **spectrum**.
The stems show sizes only: each cosine also has a phase, which a complete
recipe lists too but which these pages never need.

:::{admonition} Known result: Fourier decomposition
:class: seealso

Any signal of finite energy can be built as a sum of cosines in this way,
using a continuous range of frequencies when the signal does not repeat, and
the recipe of sizes and phases is unique. This is Fourier analysis
(Goodman, *Introduction to Fourier Optics*, section 2.1). These pages use it
without proof.
:::

## A band limit caps how rapidly the signal can oscillate

So far the spectrum has been a set of stems, one per cosine. A signal built
from many cosines packed closely in frequency has so many stems that they
merge into a filled shape, and its spectrum reads as a curve. The figure
builds three such signals, each from cosines spaced 0.002 cycles per sample
apart, with random phases and equal sizes that taper to zero just below a
cap: 0.1, 0.25 and 0.38 cycles per sample. Above its cap, each signal holds nothing.

```{code-cell} python
:tags: [hide-input]
def pair_axes(rows, height):
    """One row per signal: the signal on the left, its spectrum on the right."""
    fig, axes = plt.subplots(
        rows, 2, figsize=(9, height), constrained_layout=True, squeeze=False,
        gridspec_kw={"width_ratios": [1.5, 1]},
    )
    for left, right in axes:
        left.axhline(0, color=FAINT, lw=0.8)
        left.set_ylabel("value")
        right.set_ylabel("size")
    axes[-1, 0].set_xlabel(r"position $t$ (samples)")
    axes[-1, 1].set_xlabel("frequency (cycles per sample)")
    return fig, axes


def capped(cap, seed, spacing=0.002, taper=0.2):
    """Cosines packed `spacing` apart below `cap`, equal sizes tapering to zero at the cap."""
    f = np.arange(spacing / 2, cap, spacing)
    size = np.ones_like(f)
    if taper > 0:
        edge = f > (1 - taper) * cap
        size[edge] = np.cos(np.pi / 2 * (f[edge] - (1 - taper) * cap) / (taper * cap)) ** 2
    size *= np.sqrt(2 / np.sum(size**2))  # every signal gets the same spread of values
    phase = np.random.default_rng(seed).uniform(0, 2 * np.pi, f.size)

    def sig(t):
        return np.sum(size * np.cos(2 * np.pi * f * np.asarray(t)[..., None] + phase), axis=-1)

    return sig, f, size


t_view = np.linspace(0, 30, 2400)
fig, axes = pair_axes(3, 5.8)
for (left, right), (cap, seed) in zip(axes, ((0.1, 11), (0.25, 12), (0.38, 13))):
    sig, f, size = capped(cap, seed)
    left.plot(t_view, sig(t_view), color=pal.cyan, lw=1.5)
    left.set_xlim(0, 30)
    left.set_ylim(-3.5, 3.5)
    left.set_title(f"built from cosines below {cap}", color=pal.cyan, fontsize=11)
    f_all = np.concatenate([[0], f, [cap, 0.6]])
    rel = np.concatenate([[1], size / size.max(), [0, 0]])
    right.fill_between(f_all, 0, rel, color=pal.cyan, alpha=0.3, lw=0)
    right.plot(f_all, rel, color=pal.cyan, lw=1.4)
    right.axvline(cap, color=pal.pink, lw=1.2)
    right.text(cap + 0.015, 1.02, f"cap {cap}", color=pal.pink, fontsize=10)
    right.set_xlim(0, 0.6)
    right.set_ylim(0, 1.2)
    right.set_yticks([])
    right.set_title("its spectrum", fontsize=11)
plt.show()
```

Compare the rows from top to bottom. On the right, each spectrum is zero to
the right of its cap, and raising the cap stretches it further right. On the
left, the same change makes the signal oscillate faster. Below a cap of 0.1
cycles per sample, a rise from a low point to the next high point takes about half a
cycle at the cap, 5 samples or more, while below 0.38 it can take as little
as 1.3 samples. The right-hand edge of the spectrum sets how rapidly the
signal can oscillate. It allows fast oscillation without forcing it, since a
signal need not use every frequency below its cap.

A frequency above which a signal is known to contain nothing is its
**bandwidth** $f_{\max}$, and a signal with nothing above its bandwidth is
**band-limited**. In a plot, a band-limited signal is one whose spectrum is
zero everywhere to the right of a vertical line. The bandwidth is a bound,
so the spectrum need not reach the line: the slow curve was built below
0.38, and its fastest cosine sits near 0.31.

Now return to the three curves at the top of the page. The figure below
redraws them, each beside its spectrum, with the band below 0.38 shaded. One
extra row splits the fast curve into its two parts: the slow curve, and the
extra wave $1.2\sin(2\pi t)$ that was added to it. These spectra are measured
from each curve rather than read off a recipe, so a single cosine shows up as
a narrow peak at its frequency, as tall as the cosine is large.

```{code-cell} python
:tags: [hide-input]
T_LONG = np.arange(-256, 256, 1 / 16)
WINDOW = np.blackman(T_LONG.size)


def spectrum(values):
    """Measured spectrum of a curve given on T_LONG: a cosine of size A peaks at A."""
    pad = 4 * T_LONG.size
    size = np.abs(np.fft.rfft(values * WINDOW, n=pad)) / (WINDOW.sum() / 2)
    return np.fft.rfftfreq(pad, d=1 / 16), size


n_long = np.arange(-256, 256)
slow = signal(T_LONG)
extra = 1.2 * np.sin(2 * np.pi * T_LONG)
rows = (
    ("the slow curve", slow, pal.pink),
    ("the extra wave", extra, pal.purple),
    ("the fast curve: the slow curve plus the extra wave", slow + extra, pal.green),
    ("straight lines through the samples", np.interp(T_LONG, n_long, signal(n_long)), pal.yellow),
)
view = (T_LONG >= 0) & (T_LONG <= 11)
on_samples = np.searchsorted(T_LONG, n12)
fig, axes = pair_axes(4, 7.6)
for (left, right), (name, values, color) in zip(axes, rows):
    left.plot(T_LONG[view], values[view], color=color, lw=1.5)
    left.plot(n12, values[on_samples], "o", color=pal.cyan, ms=4)
    left.set_xlim(0, 11)
    left.set_ylim(-4, 4)
    left.set_title(name, color=color, fontsize=11)
    f, size = spectrum(values)
    right.axvspan(0, F_MAX, color=pal.pink, alpha=0.12, lw=0)
    right.axvline(F_MAX, color=pal.pink, lw=1.2)
    right.plot(f, size, color=color, lw=1.1)
    right.set_xlim(0, 1.5)
    right.set_ylim(0, 2.3)
    right.set_title("its spectrum", fontsize=11)
axes[0, 1].text(F_MAX + 0.03, 1.9, r"$f_{\max}=0.38$", color=pal.pink, fontsize=10)
plt.show()
```

Read the spectra from top to bottom. The slow curve's spectrum is twelve
narrow peaks at the positions and heights of the twelve stems built
earlier, all inside the band. The extra wave is a single cosine at 1 cycle
per sample, and every sample (cyan) lands on one of its zero crossings. The fast curve's spectrum is therefore the slow curve's
peaks plus one more at 1, outside the band. The straight lines have a corner
at every sample, and building a sharp corner takes rapid oscillation, so
their spectrum spills past the band, in small copies of the slow curve's
peaks reflected about 1 cycle per sample. Of the
three curves through the twelve samples, only the slow curve is
band-limited at 0.38.

The first requirement is to **know that the signal is band-limited, and know
its bandwidth**. Knowing it is what rules out the other two curves. This
knowledge has to come from where the signal comes from, because the samples
cannot show it: the extra wave is zero at every one of them. For a
coronagraph, the bandwidth follows from the size of the telescope pupil,
which {doc}`../theory/band-limits` derives.

:::{admonition} Why 0.38?
:class: note

Any bandwidth below 0.5 would serve these two pages. They use 0.38 because
it is the bandwidth that yippy.sinc's own interpolation is designed for, as
{doc}`../theory/kernel` explains, so the numbers here carry over. A cosine at
0.38 cycles per sample completes one cycle every $1/0.38=2.6$ samples.

The theory pages measure the same bandwidth in different units. There the
signal is a pixel's brightness as the planet moves across the sky, its
bandwidth $B$ is in cycles per $\lambda_0/D$, and the samples are
$h$ apart in $\lambda_0/D$. Converting to cycles per sample multiplies by
the spacing, $f_{\max}=B\,h$. The starter grid has $B=1$ and $h=0.38$, so
$f_{\max}=0.38$ cycles per sample: the 0.38 here and the 0.38 spacing there
are the same number for that reason ({doc}`../theory/band-limits`). The
largest spacing that keeps $f_{\max}$ below the Nyquist limit, $1/(2B)=0.5$
$\lambda_0/D$ for the starter, is called the **Nyquist spacing**, and
$\rho=2Bh$ is the spacing as a fraction of it.
:::

## What happens when the samples are too far apart?

Knowing the bandwidth is not the end of the story. To see why, look at the
samples of a single cosine at 0.7 cycles per sample, a cosine that completes
a cycle every 1.4 samples, with the cosine itself hidden.

```{code-cell} python
:tags: [hide-input]
n_a = np.arange(0, 11)
fig, ax = plt.subplots(figsize=(7.5, 2.2), constrained_layout=True)
ax.axhline(0, color=FAINT, lw=0.8)
ax.plot(n_a, np.cos(2 * np.pi * 0.7 * n_a), "o", color=pal.cyan, ms=6)
ax.set_xlim(-0.3, 10.3)
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel(r"position $t$ (samples)")
ax.set_ylabel("value")
ax.set_title("samples of a cosine at 0.7 cycles per sample", fontsize=11)
plt.show()
```

```{admonition} Predict before reading on
:class: tip
Join the dots by eye with the smoothest curve you can. Does it look like a
cosine faster than 0.7 cycles per sample, the same, or slower? Roughly how
many samples does your curve take per cycle?
```

In the animation the cosine (gray) speeds up from 0.05 to 0.95 cycles per
sample while the samples (cyan) stay one unit apart. The pink curve is the
slowest cosine that passes through every sample. Watch the moment the
frequency passes 0.5: the pink curve leaves the gray one and starts to slow
down while the true cosine keeps speeding up.

```{code-cell} python
:tags: [hide-input]
# An animation rather than a strip: at f = 0.5 the slowest cosine through the
# samples leaves the true cosine and starts slowing down, a reversal of
# direction that a strip of stills does not carry. The still follows.
sweep = np.concatenate([np.full(2, 0.05), np.linspace(0.05, 0.95, 37), np.full(4, 0.95)])
t_a = np.linspace(0, 10, 1200)
fig, ax = plt.subplots(figsize=(7.5, 2.8), constrained_layout=True)
(true_line,) = ax.plot([], [], color=SCENERY, lw=1.1)
(alias_line,) = ax.plot([], [], color=pal.pink, lw=2)
(dots,) = ax.plot([], [], "o", color=pal.cyan, ms=5)
ax.set_xlim(0, 10)
ax.set_ylim(-1.3, 1.7)
ax.set_xlabel(r"position $t$ (samples)")
ax.set_ylabel("value")
readout = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=10)


def draw(fig, f):
    seen = min(f, 1 - f)
    true_line.set_data(t_a, np.cos(2 * np.pi * f * t_a))
    alias_line.set_data(t_a, np.cos(2 * np.pi * seen * t_a))
    dots.set_data(n_a, np.cos(2 * np.pi * f * n_a))
    readout.set_text(f"true cosine: {f:.2f} cycles/sample    samples look like: {seen:.2f}")


anim = ep.render.animate(fig, draw, sweep, fps=6)
clip(anim)
```

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 4, figsize=(10, 2.4), sharey=True, constrained_layout=True)
for ax, f in zip(axes, (0.25, 0.45, 0.55, 0.7)):
    seen = min(f, 1 - f)
    ax.plot(t_a, np.cos(2 * np.pi * f * t_a), color=SCENERY, lw=1.1)
    ax.plot(t_a, np.cos(2 * np.pi * seen * t_a), color=pal.pink, lw=1.8)
    ax.plot(n_a, np.cos(2 * np.pi * f * n_a), "o", color=pal.cyan, ms=4)
    ax.set_title(f"true {f}, looks like {seen:.2f}", fontsize=10)
    ax.set_xlabel(r"$t$ (samples)")
axes[0].set_ylabel("value")
plt.show()
```

```{dropdown} Answer to the prediction
Slower: about 3.3 samples per cycle, a cosine at $1-0.7=0.3$ cycles per
sample (right panel of the still). The samples of the fast cosine are exactly
the samples of the slow one.
```

Below 0.5 cycles per sample the slowest cosine through the samples is the
true one, so the pink and gray curves coincide. Between 0.5 and 1 a slower
cosine, at $1-f$, fits the same samples. The reason is a property of cosines
at whole-number positions. For any integer $n$,

$$
\cos\!\big(2\pi(1-f)\,n\big)=\cos(2\pi n-2\pi f n)=\cos(-2\pi f n)=\cos(2\pi f n).
$$

The first step only expands the bracket. The second drops $2\pi n$, a whole
number of turns, which does not change a cosine. The third uses the symmetry
of the cosine, $\cos(-x)=\cos x$. In words: at the sample positions, a
cosine at $f$ and a cosine at $1-f$ take the same values, where 1 is the
sampling rate of one sample per unit. Seeing a fast cosine as a slow one is
called **aliasing**, and nothing in the samples reveals it.

The fast curve of the first figure is the same effect one step further: its
extra wave at 1 cycle per sample looks, at the samples, like a wave at
$1-1=0$ cycles per sample, and since it is zero at every sample it adds
nothing the samples can see. Above 1 cycle per sample the pattern repeats,
and the signal shifts in phase as well as frequency, but the demonstrations
here need only the range up to 1.

## The Nyquist limit: more than two samples per cycle

The aliasing rule pairs every frequency $f$ with a twin at $1-f$ that gives
the same samples. The figure plots the frequency the samples appear to show
against the true frequency, with the four cosines of the still marked.

```{code-cell} python
:tags: [hide-input]
f_true = np.linspace(0, 1, 401)
fig, ax = plt.subplots(figsize=(7.5, 3.0), constrained_layout=True)
ax.axvspan(0.5, 1, color=FAINT, alpha=0.35, lw=0)
ax.plot([0, 1], [0, 1], color=FAINT, lw=1, ls="--")
ax.text(0.56, 0.6, "if the samples showed the truth", color=SCENERY, fontsize=9, rotation=45,
        rotation_mode="anchor", transform_rotates_text=True)
ax.plot(f_true, np.minimum(f_true, 1 - f_true), color=pal.pink, lw=2)
ax.axvline(0.5, color=SCENERY, lw=1)
ax.text(0.49, 0.03, "Nyquist limit 0.5", color=SCENERY, fontsize=9, ha="right")
for f, ha, dx in ((0.25, "right", -8), (0.45, "right", -8), (0.55, "left", 8), (0.7, "left", 8)):
    seen = min(f, 1 - f)
    ax.plot(f, seen, "o", color=pal.cyan, ms=6)
    ax.annotate(f"{f} looks like {seen:.2f}", (f, seen), xytext=(dx, 4),
                textcoords="offset points", ha=ha, va="bottom", color=pal.cyan, fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.75)
ax.set_xlabel("true frequency (cycles per sample)")
ax.set_ylabel("frequency the\nsamples show")
plt.show()
```

Up to 0.5 cycles per sample the line rises along the dashed diagonal: each
cosine appears as itself, and different cosines appear different. Past 0.5
it folds back, so every frequency the samples can show below 0.5 is shared
by two true frequencies, $f$ and $1-f$. The fold sits at $f=0.5$, two
samples per cycle, and this dividing line is the **Nyquist limit**.

Aliasing alone does not doom a signal, because a band limit can rule twins
out, the way it ruled out the fast curve's extra wave. It can only rule out a
twin that lies outside the band. The figure tests three bandwidths, 0.38,
0.5 and 0.62 cycles per sample, with the samples one unit apart as always.
Each signal (gray) is drawn beside its spectrum, together with the spectrum
of its twins (pink): every cosine moved from $f$ to $1-f$.

```{code-cell} python
:tags: [hide-input]
TRUE = ep.style.neutral(0.6)
t_q = np.linspace(0, 12, 1200)
n_q = np.arange(0, 13)


def zero_at_samples(cap):
    """A change, inside a band reaching `cap`, that is zero at every sample."""
    if cap < 0.5:
        return None
    if cap == 0.5:
        return lambda t: 0.9 * np.sin(np.pi * t)
    return lambda t: 0.9 * (np.cos(2 * np.pi * 0.55 * t - 1) - np.cos(2 * np.pi * 0.45 * t + 1))


titles = {
    0.38: "bandwidth 0.38: one curve through the samples",
    0.5: "bandwidth 0.5: a second curve fits",
    0.62: "bandwidth 0.62: a second curve fits",
}
fig, axes = pair_axes(3, 6.2)
for (left, right), (cap, seed) in zip(axes, ((0.38, 21), (0.5, 22), (0.62, 23))):
    sig, f, size = capped(cap, seed, taper=0)
    left.plot(t_q, sig(t_q), color=TRUE, lw=1.4)
    change = zero_at_samples(cap)
    if change is not None:
        left.plot(t_q, sig(t_q) + change(t_q), color=pal.pink, lw=1.6)
    left.plot(n_q, sig(n_q), "o", color=pal.cyan, ms=4.5)
    left.set_xlim(0, 12)
    left.set_ylim(-4, 4)
    left.set_title(titles[cap], fontsize=11)

    band_f = np.array([0, cap, cap])
    band = np.array([1, 1, 0])
    right.fill_between(band_f, 0, band, color=TRUE, alpha=0.35, lw=0)
    right.plot(band_f, band, color=TRUE, lw=1.3)
    right.fill_between(1 - band_f, 0, band, color=pal.pink, alpha=0.3, lw=0)
    right.plot(1 - band_f, band, color=pal.pink, lw=1.3)
    right.axvline(0.5, color=SCENERY, lw=1, ls="--")
    right.set_xlim(0, 1)
    right.set_ylim(0, 1.25)
    right.set_yticks([])
    right.set_title("its spectrum and its twins", fontsize=11)
axes[0, 1].text(0.51, 1.08, "Nyquist 0.5", color=SCENERY, fontsize=9)
axes[0, 1].text(0.2, 1.08, "signal", color=TRUE, fontsize=9, ha="center")
axes[0, 1].text(0.8, 1.08, "twins", color=pal.pink, fontsize=9, ha="center")
axes[2, 1].text(0.5, 1.08, "both allowed", fontsize=9, ha="center")
plt.show()
```

In the top row the band stops at 0.38, so its twins fill 0.62 to 1 and never
touch it. The band limit rules every twin out, and no second curve in the
band fits the samples. In the bottom row the band reaches 0.62 and its twins
reach down to 0.38, so between 0.38 and 0.62 a cosine and its twin are both
allowed. The pink curve swaps a cosine at 0.45 for its twin at 0.55: it is a
different curve with the same bandwidth, yet it passes through every
sample.

The middle row is the edge case. The band and its twins meet at 0.5,
where a cosine is its own twin and can sit on its zero crossings, as the pink
curve shows.

The second requirement is therefore to **place the samples close enough that
the bandwidth stays below the Nyquist limit**, strictly below it, so that
the band and its twins never meet.

One number says how much of that allowance a grid uses, the **oversampling
factor** $\rho=2f_{\max}$, the bandwidth as a fraction of the Nyquist limit.
Despite its name, a smaller $\rho$ means more samples per cycle than
needed, and $\rho=1$ is the limit itself. The test signal has
$\rho=2\times0.38=0.76$: at least 2.6 samples per cycle of any cosine it
can contain. The three rows of the figure above have
$\rho=0.76$, 1 and 1.24.

:::{admonition} Known result: the sampling theorem
:class: seealso

Among signals with no cosine at or above 0.5 cycles per sample, the samples
determine the signal completely, so whenever $\rho<1$ exactly one such curve
passes through them. The inequality must be strict: a cosine at exactly 0.5
cycles per sample can be sampled at its zero crossings and vanish from the
samples. The result is credited to Whittaker (1915), Nyquist (1928),
Kotelnikov (1933) and Shannon (1949); Goodman's *Introduction to Fourier
Optics*, section 2.4, gives the two-dimensional version used in optics. The
{doc}`next page <sinc-interpolation>` shows how the values between samples
are computed, which is also how the theorem is proved.
:::

These requirements assume noiseless samples on an evenly spaced grid. Energy
above the bandwidth, however small, would alias into the samples, and an
uneven grid would break the whole-number argument above.

## Try it

yippy.sinc makes the same two checks for every delivered grid, once along each
axis: the bandwidth comes from the telescope pupil, and the grid spacing then
sets $\rho$. The exercises practice those checks.

1. A signal has a bandwidth of 0.3 cycles per sample. What is $\rho$, and how
   many samples per cycle does a cosine at the bandwidth get?
   ```{dropdown} Hint
   Use $\rho=2f_{\max}$, and samples per cycle $=1/f$.
   ```
   ```{dropdown} Answer
   $\rho=0.6$, and $1/0.3=3.3$ samples per cycle.
   ```
2. Predict, then change the tuple `(0.25, 0.45, 0.55, 0.7)` in the still's
   cell to end with `0.85`. Which slower cosine passes through the samples?
   ```{dropdown} Hint
   At the sample positions a cosine at $f$ matches one at $1-f$.
   ```
   ```{dropdown} Answer
   0.15 cycles per sample, a cosine that repeats every 6.7 samples.
   ```
3. A grid has $\rho=0.5$. How many samples does a cosine at the bandwidth get
   per cycle, and is the grid closer to or further from aliasing than the
   test signal's?
   ```{dropdown} Hint
   First recover $f_{\max}$ from $\rho$.
   ```
   ```{dropdown} Answer
   $f_{\max}=0.25$, so 4 samples per cycle, further from aliasing than the
   test signal's 2.6.
   ```

## What this page established

:::{admonition} What perfect interpolation needs, so far
:class: important

1. **A band limit.** The signal has no cosine above a known bandwidth
   $f_{\max}$.
2. **Dense enough samples.** More than two samples per cycle at the
   bandwidth, $\rho=2f_{\max}<1$.
3. *To be found on the next page: how to combine the samples.*
:::

With both requirements met, exactly one curve with no cosine at or above
0.5 cycles per sample passes through the samples. The next page builds the
procedure that finds it, starting from the single shape that each sample
contributes.

Next: {doc}`sinc-interpolation`.
