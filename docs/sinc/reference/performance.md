---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  render_markdown_format: myst
---

# What does a read cost?

A coronagraph yield code asks for a planet image at every orbit step of every
trial, so the cost of one read decides whether reading between grid nodes is
affordable. This page reports what version 0.1 costs for the physicaloptix
starter delivery on one laptop CPU, what a batch of $10^4$ positions costs,
and how that compares with yippy's current blend interpolation on the same
positions. Every measured number comes from `docs/scripts/benchmark.py`, which
writes `_static/validation/benchmark.json` beside the validation figures. Rerun
it on your own machine before relying on a number.

A read here is one 128 by 128 pixel image. Timings are *steady* when the code
has already been compiled by `jax.jit` and run once, *first call* when they
include that compile, and *resident* memory is what the process holds in RAM
(resident set size, RSS).


```{code-cell} python
:tags: [remove-cell]
import math
import sys

sys.path.insert(0, "../scripts")
from IPython.display import Markdown, display
from validation_tables import load, sci

b = load("benchmark.json")
nodes = load("validation.json")["training_nodes"]
m = b["machine"]
f32, f64, yp = b["float32"], b["float64"], b["yippy_blend"]
ms = lambda s: f"{1e3 * s:.1f} ms"
per = lambda r: f"{1e3 * r['seconds_per_read']:.1f} ms"
bs32_first = f32["batch_small_first_call_seconds"]
```

## Where the numbers were taken

```{code-cell} python
:tags: [remove-input]
display(Markdown(
    f"- Machine: {m['chip']}, {m['total_number_of_cores']} cores, {m['memory']}.\n"
    f"- Software: Python {m['python']}, JAX {m['jax']} on the {m['backend']} backend.\n"
    f"- Delivery: {nodes} float32 nodes of 128 by 128 pixels, {f32['resident_mib']:.0f} MiB resident "
    f"({f64['resident_mib']:.0f} MiB if stored as float64).\n"
    f"- Load: the machine was shared with other jobs (load average "
    f"{m['load_average_at_start'][0]:.1f} at the start and {b['load_average_after_reads'][0]:.1f} "
    f"after the reads), so treat the timings as upper bounds on an idle machine. The float32 and "
    f"float64 readers were timed in alternating rounds, and float64 came out faster here, "
    f"which is within what contention and memory traffic can do and is not a claim about the kernel.\n"
    f"- GPU: untested. No GPU number exists for version 0.1."
))
```

## Cost of one read and of a batch

```{code-cell} python
:tags: [remove-input]
bs32, bs64 = f32["batch_small"], f64["batch_small"]
rows = [
    ("Admission, first call in the process", f"{b['admission_first_call_seconds']:.1f} s", "includes JAX start-up work"),
    ("Admission, later calls", f"{b['admission_later_calls_median_seconds']:.1f} s",
     f"minimum {b['admission_later_calls_min_seconds']:.1f} s over 4 calls"),
    ("Compile plus first single read, float32", f"{f32['single_first_call_seconds']:.2f} s", ""),
    ("Steady single read, float32", ms(f32["single_steady_median_seconds"]),
     f"range {ms(f32['single_steady_min_seconds'])} to {ms(f32['single_steady_max_seconds'])}"),
    ("Steady single read, float64 storage", ms(f64["single_steady_median_seconds"]),
     f"range {ms(f64['single_steady_min_seconds'])} to {ms(f64['single_steady_max_seconds'])}"),
    ("`read_batch`, 200 positions, float32", ms(bs32["seconds_per_read_median"]) + " per read",
     f"minimum {ms(bs32['seconds_per_read_min'])}, first call {bs32_first:.1f} s"),
    ("`read_batch`, 200 positions, float64 storage", ms(bs64["seconds_per_read_median"]) + " per read",
     f"minimum {ms(bs64['seconds_per_read_min'])}"),
    ("`read_batch`, 10 000 positions, float32", per(f32["batch_10000"]) + " per read",
     f"{f32['batch_10000']['seconds']:.0f} s total, one run, compile included"),
    ("`read_batch`, 10 000 positions, float64 storage", per(f64["batch_10000"]) + " per read",
     f"{f64['batch_10000']['seconds']:.0f} s total, one run, compile included"),
    ("yippy blend, single read", ms(yp["single_steady_median_seconds"]), ""),
    (f"yippy blend, {yp['batch']['positions']} positions in chunks of {yp['batch']['chunk']}",
     per(yp["batch"]) + " per read", f"{yp['batch']['seconds']:.0f} s total"),
]
display(Markdown("| Quantity | Result | Note |\n|---|---|---|\n" + "\n".join(f"| {a} | {r} | {n} |" for a, r, n in rows)))
```

```{code-cell} python
:tags: [remove-input]
PIXELS = 128 * 128
TAPS = 32 * 32
single_ratio = f32["single_steady_median_seconds"] / yp["single_steady_median_seconds"]
batch_ratio = bs32["seconds_per_read_median"] / yp["batch"]["seconds_per_read"]
macs = PIXELS * TAPS
flops_sinc = 2 * macs
fft_flops = {n: 4 * 2 * 5 * n * n * math.log2(n * n) for n in (128, 256)}
traffic = macs * 4  # float32 node values gathered once per tap and pixel, in bytes
display(Markdown(
    f"A steady single read took {ms(f32['single_steady_median_seconds'])} with float32 storage "
    f"and {ms(f64['single_steady_median_seconds'])} with float64 storage, and the yippy blend "
    f"took {ms(yp['single_steady_median_seconds'])}. The blend is faster than the sinc read by a "
    f"factor of {single_ratio:.0f} (float32). Over {bs32['positions']} positions in a batch the "
    f"sinc read took {ms(bs32['seconds_per_read_median'])} per read and the blend over "
    f"{yp['batch']['positions']} positions took {per(yp['batch'])}, a factor of {batch_ratio:.0f}. "
    f"The $10^4$ batch in the table is not used for this comparison because its time includes "
    f"the compile. For scale, $10^6$ reads at the steady float32 time need "
    f"{1e6 * f32['single_steady_median_seconds'] / 3600:.0f} hours on one core, an illustration "
    f"and not the requirement of any particular yield code."
))
```

## Is the sinc read limited by arithmetic or by memory?

```{code-cell} python
:tags: [remove-input]
t_read = f32["single_steady_median_seconds"]
display(Markdown(
    f"One read sums {TAPS} products for each of {PIXELS} pixels, {macs:.1e} multiply-adds or "
    f"{flops_sinc:.1e} floating-point operations. It also gathers {macs:.1e} float32 node values, "
    f"{traffic / 1e6:.0f} MB. In {1e3 * t_read:.0f} ms that is {flops_sinc / t_read / 1e9:.2f} "
    f"Gflop/s of arithmetic and {traffic / t_read / 1e9:.1f} GB/s of reads. "
    "The blend does four forward and four inverse FFTs of 5 N log2 N operations each, "
    f"{fft_flops[128]:.1e} operations if the images are padded to 128 by 128 and "
    f"{fft_flops[256]:.1e} if padded to 256 by 256 (the padding is not recorded here), "
    "so the two methods do about the same amount of arithmetic, while the blend runs "
    f"{single_ratio:.0f} times faster. The operation count therefore does not explain the "
    "difference. Neither rate above is compared with this chip's peak, because the "
    "benchmark did not measure one, so this page cannot say whether the sinc read is limited "
    "by arithmetic, by gathering the nodes from memory or by the compiled code. Finding out "
    "is open work."
))
```

## Memory

```{code-cell} python
:tags: [remove-input]
display(Markdown(
    f"- Peak resident memory of the process after loading the delivery: {b['after_load_rss_mib']:.0f} MiB.\n"
    f"- After admission: {b['admission_peak_rss_mib']:.0f} MiB.\n"
    f"- After the float32 reads, including the $10^4$ batch: {f32['batch_10000']['peak_rss_mib_after']:.0f} MiB.\n"
    f"- After the float64 reads: {f64['batch_10000']['peak_rss_mib_after']:.0f} MiB.\n"
    f"- A batch of $10^4$ returns {f32['batch_10000']['output_gib']:.2f} GiB of float64 images, "
    f"so the output ceiling of its `MemoryBudget` must exceed that, and the benchmark sets "
    f"2 GiB (`MemoryBudget` has no defaults). Chunk larger campaigns."
))
```

The peak figures are the process's peak resident set size after each stage, so
they only rise. The resident delivery itself is the Delivery line under "Where the numbers were
taken", and the output of a batch dominates everything else.

## What does the prefiltered profile cost?

The prefiltered profile ({doc}`../practice/prefiltered-read`) sums 49 or 64
filtered images where the raw read sums 1,024. No benchmark of it on the
starter delivery exists, because that delivery is sampled at the raw limit
and the prefiltered profile needs a denser grid. The cell below instead times
both profiles on the toy coronagraph of the theory pages, a quadrant of 41 by
41 nodes at a spacing of 0.32 $\lambda_0/D$ with 96 by 96 pixel images in
float64 storage. It runs when this page is built, so its numbers belong to
the build machine and vary with its load.

```{code-cell} python
:tags: [hide-input]
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import yippy.sinc as sc
from yippy.sinc.kernel import PrefilteredKernelProfile
from yippy.sinc.reader import Interpolator, PrefilteredReader, read_batch

import toy_coronagraph as tc

delivery = tc.quadrant_delivery(spacing=0.32, nodes=41, domain_half=5.0)
positions = jnp.asarray(np.random.default_rng(5).uniform(-5.0, 5.0, (64, 2)))
product = jnp.zeros(64, dtype=jnp.int32)
tiling = sc.Tiling(query_chunk=8, pixel_tile=96 * 96)
batch = eqx.filter_jit(lambda reader, q, p: read_batch(reader, q, p).intensity)


def steady(function, *arguments, rounds=5):
    """Return the fastest of several timed calls after one warm-up call."""
    jax.block_until_ready(function(*arguments))
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        jax.block_until_ready(function(*arguments))
        best = min(best, time.perf_counter() - start)
    return best


def build(profile, kind):
    """Admit the toy delivery for a profile and time the reader's construction."""
    start = time.perf_counter()
    grid = sc.admit_grid(**delivery, profile=profile).require()
    reader = kind(grid, delivery["domain"], profile, sc.ReadPolicy(), tiling,
                  delivery["memory"])
    return reader, time.perf_counter() - start


raw, raw_load = build(sc.KernelProfile(), Interpolator)
rows = [("Raw, 32 taps", raw_load, steady(batch, raw, positions, product) / 64)]
readers = {}
for taps in (7, 8):
    profile = PrefilteredKernelProfile(taps=taps)
    readers[taps], load = build(profile, PrefilteredReader)
    rows.append((f"Prefiltered, {taps} taps", load,
                 steady(batch, readers[taps], positions, product) / 64))

renderer = sc.SceneRenderer(readers[7], 0, positions)
render = eqx.filter_jit(lambda r, q, f: r.render_points(q, f).intensity)
scene_time = steady(render, renderer, positions, jnp.ones(64))
table = "| Profile | Admission and load | Steady read, batch of 64 | Raw / this |\n|---|---|---|---|\n"
table += "\n".join(
    f"| {name} | {load:.1f} s | {1e3 * read:.2f} ms per read | {rows[0][2] / read:.0f} |"
    for name, load, read in rows
)
display(Markdown(
    table + f"\n\nRendering the 64 positions as one scene with 7 taps took "
    f"{1e3 * scene_time:.2f} ms, against {1e3 * 64 * rows[1][2]:.1f} ms for the 64 "
    f"separate reads, over {renderer.nodes.shape[0]} touched nodes."
))
```

The gain is smaller than the ratio of image counts, 1,024 to 49, because
both reads also pay for the result and its diagnostics, and because a
quadrant read gathers mirrored images one by one. Admission and load take
longer for the prefiltered profile: they design the filter and build the
coefficient images, once per grid. The accuracy given up for the speed is on
{doc}`../practice/prefiltered-read`.

## What to take from this

```{code-cell} python
:tags: [remove-input]
display(Markdown(
    f"A steady read costs {ms(f64['single_steady_median_seconds'])} to "
    f"{ms(f32['single_steady_median_seconds'])} on a loaded CPU, and it is {single_ratio:.0f} "
    "times slower than yippy's blend. The measured single-read times ranged from "
    f"{ms(f64['single_steady_min_seconds'])} to {ms(f32['single_steady_max_seconds'])} across "
    "rounds, which is the load. Batches of positions are the unit to optimize, and a GPU "
    "might help most there, but no GPU measurement exists (GPU untested)."
))
```

Next: the {doc}`../validation` page shows what those reads are worth.
