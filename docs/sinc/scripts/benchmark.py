"""Benchmark yippy.sinc reads on the delivery and compare with yippy's blend backend.

Measures, on the machine that runs it: admission time, compile time, steady-state
single read, ``read_batch`` over 1e4 positions, peak resident memory, float32
versus float64 storage, and yippy's current inverse-distance blend on the same
positions. Writes ``docs/_static/validation/benchmark.json`` beside the figures.

Usage::

    python docs/scripts/benchmark.py DELIVERY [--no-blend --out NAME]

``--no-blend`` skips the yippy comparison (for machines without yippy) and
``--out`` names the JSON file written beside the figures.

Timings are wall-clock seconds with the device synchronized. Peak memory is the
process peak resident set size after each stage (monotone, so a later stage
reports the largest earlier stage unless it exceeds it).
"""

import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import yippy.sinc as sc  # noqa: E402
from yippy.sinc.reader import Interpolator, read_batch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_figures as vf  # noqa: E402

CELL = np.array([[3.8, 5.32], [4.18, 5.70]])
GIB = 2**30


def rss_mib():
    """Peak resident set size of this process so far, in MiB."""
    scale = 1 if platform.system() == "Darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2**20


def machine():
    """Describe the machine and software the timings were taken on."""
    info = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        jax=jax.__version__,
        backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],
        load_average_at_start=list(os.getloadavg()),
    )
    try:
        text = subprocess.run(
            ["system_profiler", "SPHardwareDataType"], capture_output=True, text=True
        ).stdout
        for line in text.splitlines():
            key, _, value = line.strip().partition(": ")
            if key in ("Chip", "Total Number of Cores", "Memory"):
                info[key.lower().replace(" ", "_")] = value
    except OSError:
        pass
    return info


def sync(value):
    """Block until every array in ``value`` is computed, then return it."""
    jax.block_until_ready(jax.tree_util.tree_leaves(value))
    return value


def clock(fn, repeat):
    """Wall-clock seconds of ``repeat`` synchronized calls to ``fn``."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        sync(fn())
        times.append(time.perf_counter() - start)
    return times


def positions(count, seed):
    """Seeded random source positions [count, 2] inside the served cell."""
    rng = np.random.default_rng(seed)
    return CELL[0] + rng.random((count, 2)) * (CELL[1] - CELL[0])


def make_reader(grid, arguments):
    """An Interpolator with room for a 1e4-position batch of output."""
    memory = sc.MemoryBudget(2 * GIB, 2 * GIB, GIB, 8 * GIB)
    return Interpolator(
        grid,
        arguments["domain"],
        sc.KernelProfile(),
        sc.ReadPolicy(),
        sc.Tiling(pixel_tile=2048),
        memory,
    )


def bench_readers(grids, arguments, rounds=3, small=200):
    """Time reads from each storage precision, alternating them round by round.

    Alternation spreads load from other jobs on a shared machine evenly over
    the precisions; each figure is the median and the minimum over rounds.
    The 1e4 batch runs once per precision and includes about a second of compile.
    """
    readers = {k: make_reader(g, arguments) for k, g in grids.items()}
    q1 = jnp.asarray(positions(1, 1)[0])
    qs = jnp.asarray(positions(small, 2))
    ps = jnp.zeros(small, dtype=jnp.int32)
    product = jnp.int32(0)
    out, jitted = {}, {}
    for label, reader in readers.items():
        one = eqx.filter_jit(reader.read)
        batch = eqx.filter_jit(read_batch)
        start = time.perf_counter()
        sync(one(q1, product))
        first = time.perf_counter() - start
        start = time.perf_counter()
        sync(batch(reader, qs, ps))
        small_first = time.perf_counter() - start
        jitted[label] = (one, batch)
        out[label] = dict(
            storage=label,
            resident_mib=grids[label].intensity.nbytes / 2**20,
            single_first_call_seconds=first,
            batch_small_first_call_seconds=small_first,
            _single=[],
            _small=[],
        )
    for _ in range(rounds):
        for label, reader in readers.items():
            one, batch = jitted[label]
            out[label]["_single"] += clock(partial(one, q1, product), 20)
            out[label]["_small"] += clock(partial(batch, reader, qs, ps), 1)
    for entry in out.values():
        single, small_runs = entry.pop("_single"), entry.pop("_small")
        entry.update(
            single_steady_median_seconds=statistics.median(single),
            single_steady_min_seconds=min(single),
            single_steady_max_seconds=max(single),
            batch_small=dict(
                positions=small,
                rounds=rounds,
                seconds_per_read_median=statistics.median(small_runs) / small,
                seconds_per_read_min=min(small_runs) / small,
            ),
        )
    for label, reader in readers.items():
        count = 10000
        q = jnp.asarray(positions(count, 3))
        p = jnp.zeros(count, dtype=jnp.int32)
        batch = eqx.filter_jit(read_batch)
        start = time.perf_counter()
        sync(batch(reader, q, p))
        total = time.perf_counter() - start
        out[label]["batch_10000"] = dict(
            positions=count,
            seconds=total,
            seconds_per_read=total / count,
            output_gib=count * 128 * 128 * 8 / GIB,
            load_average_after=list(os.getloadavg()),
            peak_rss_mib_after=rss_mib(),
        )
    return out


def bench_blend(root, count_batch=10000, chunk=100):
    """Time yippy's blend interpolation on the same kind of positions."""
    from lod_unit import lod

    from yippy._offjax import OffJAX

    off = OffJAX(
        root / "mono-000",
        "offax_psf.fits",
        "offax_psf_offset_list.fits",
        0.25 * lod,
        False,
        False,
    )
    xy = positions(count_batch, 3)

    def call(part):
        return off.create_psfs(part[:, 0], part[:, 1])

    sync(call(xy[:1]))
    single = clock(lambda: call(xy[:1]), 20)
    sync(call(xy[:chunk]))
    start = time.perf_counter()
    for i in range(0, count_batch, chunk):
        sync(call(xy[i : i + chunk]))
    total = time.perf_counter() - start
    return dict(
        single_steady_median_seconds=statistics.median(single),
        batch=dict(
            positions=count_batch,
            chunk=chunk,
            seconds=total,
            seconds_per_read=total / count_batch,
        ),
        peak_rss_mib_after=rss_mib(),
    )


def main(root, out_dir, name="benchmark.json", blend=True):
    """Run every benchmark and write the JSON beside the figures."""
    result = dict(machine=machine())
    arguments, *_ = vf.load(root)
    result["after_load_rss_mib"] = rss_mib()
    times = []
    for _ in range(4):
        start = time.perf_counter()
        admission = sc.admit_grid(**arguments)
        times.append(time.perf_counter() - start)
    result["admission_first_call_seconds"] = times[0]
    result["admission_later_calls_median_seconds"] = statistics.median(times[1:])
    result["admission_later_calls_min_seconds"] = min(times[1:])
    result["admission_peak_rss_mib"] = rss_mib()
    grid32 = admission.require()
    grid64 = eqx.tree_at(
        lambda g: g.intensity, grid32, jnp.asarray(grid32.intensity, dtype=jnp.float64)
    )
    result.update(bench_readers({"float32": grid32, "float64": grid64}, arguments))
    result["load_average_after_reads"] = list(os.getloadavg())
    if blend:
        result["yippy_blend"] = bench_blend(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    args = sys.argv[1:]
    name = args[args.index("--out") + 1] if "--out" in args else "benchmark.json"
    main(
        Path(args[0]),
        here.parent / "_static" / "validation",
        name,
        blend="--no-blend" not in args,
    )
