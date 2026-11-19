---
file_format: mystnb
kernelspec:
  name: python3
---

# Reproducible stencils

A read gathers the 32 nodes of its stencil along each axis
({doc}`../foundations/sinc-interpolation`). Write $s$ for the query, $o$ for the
lattice origin and $h$ for the node spacing, all in $\lambda_0/D$, and
$t=(s-o)/h$ for the query measured in node spacings (the $t$ of the foundations
pages). The **cell** of the query is the interval between nodes $\lfloor t\rfloor$
and $\lfloor t\rfloor+1$, and the stencil runs from node `floor(t) - 15` to
node `floor(t) + 16`, 16 nodes at or below the query and 16 above.

```{code-cell} python
:tags: [remove-cell]
import math
from fractions import Fraction

import eyepiece as ep
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from yippy.sinc.kernel import exact_quotient_floor_v1

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette
```

```{code-cell} python
:tags: [hide-input]
FAINT = ep.style.neutral(0.25)
SCENERY = ep.style.neutral(0.45)
t_demo = 7.567
cell = math.floor(t_demo)
nodes = np.arange(-10, 26)
in_stencil = (nodes >= cell - 15) & (nodes <= cell + 16)
fig, ax = plt.subplots(figsize=(9, 2.2), constrained_layout=True)
ax.axvspan(cell, cell + 1, color=pal.pink, alpha=0.25, lw=0)
ax.axhline(0, color=FAINT, lw=0.8)
ax.plot(nodes[~in_stencil], np.zeros((~in_stencil).sum()), "o", color=SCENERY, ms=5)
ax.plot(nodes[in_stencil], np.zeros(in_stencil.sum()), "o", color=pal.cyan, ms=6)
ax.plot([t_demo], [0], "|", color=pal.pink, ms=22, mew=2.5)
ax.set_ylim(-1, 1.6)
ax.set_yticks([])
ax.set_xlabel("node index")
ax.set_title(
    f"query t = {t_demo}: cell {cell}, stencil = nodes {cell - 15} to {cell + 16}",
    fontsize=11,
)
ax.text(cell + 0.5, 0.55, f"cell {cell}", color=pal.pink, ha="center", fontsize=10)
ax.text(cell - 15, 0.55, "first tap", color=pal.cyan, ha="center", fontsize=9)
ax.text(cell + 16, 0.55, "last tap", color=pal.cyan, ha="center", fontsize=9)
plt.show()
```

The figure draws one axis: the 32 taps (cyan nodes) around a query (pink tick)
that lies in the shaded cell. Move the query across a cell edge and the whole
stencil shifts by one node, which is why the floor decides which nodes are read.

Whether those nodes exist decides whether the read is served or refused. If two
ways of running the same code (eager Python, meaning line by line without
compilation, `jax.jit` and `jax.vmap`) computed different floors for the same
query, the same read could be accepted in one and refused in another, and
receipts would not be reproducible. This page shows that the obvious formula does
exactly that, and what yippy.sinc does instead.

```{admonition} Predict before reading on
:class: tip
With spacing `h = 0.38`, what is the floor of the exact quotient of the stored
number 1.9 by the stored number 0.38, and what does Python's
`math.floor(1.9 / h)` return? Both are `float64` numbers (doubles), and the
literal 1.9 is the double nearest $5\times0.38$.
```

```{dropdown} Answer
The exact floor is 4, and Python returns 5. The exact quotient of the two
stored doubles is just below 5: the double nearest 1.9 is slightly smaller than
five times the double nearest 0.38. Python's division is correctly rounded, but
the correctly rounded quotient lands on the integer 5.
```

```{code-cell} python
h = 0.38
for q in (1.9, 1.14):
    exact = math.floor(Fraction(q) / Fraction(h))
    python = math.floor(q / h)
    jitted = int(jax.jit(lambda a: jnp.floor(a / h))(q))
    rule = int(jax.jit(lambda a: exact_quotient_floor_v1(a, 0.0, h))(q))
    print(f"q = {q}: exact {exact}, Python {python}, jax.jit {jitted}, yippy.sinc {rule}")
print(f"1.14 / h = {1.14 / h!r}, but 1.14 * (1 / h) = {1.14 * (1 / h)!r}")
```

Two different failures appear. At 1.9 both Python and `jax.jit` round the
quotient up to an integer the exact quotient does not reach. At 1.14, Python
gets the exact answer and `jax.jit` does not, because compiled code on CPU
replaces division by a constant with multiplication by its reciprocal, which
rounds twice (the cell below shows it: `1.14 * (1 / h)` is exactly 3.0). The second failure is the dangerous one, because it makes the
answer depend on how the code was run.

## Which method puts each query in the right cell?

The figure walks through the doubles (float64 numbers) nearest the cell edge at
$3\times0.38$, one unit in the last place (ULP) at a time, and marks the floor
each method computes. Exact arithmetic switches from 2 to 3 at one double. At
this edge Python's quotient agrees with it, and the `jax.jit` quotient switches
one double too early. The edge at 1.9 above is one where Python is also wrong, so
the naive formulas fail at different edges.

```{code-cell} python
:tags: [hide-input]
edge = 3 * h
queries = [edge]
for _ in range(6):
    queries.insert(0, math.nextafter(queries[0], -math.inf))
    queries.append(math.nextafter(queries[-1], math.inf))
offsets = np.arange(-6, 7)
rows = {
    "exact": [math.floor(Fraction(q) / Fraction(h)) for q in queries],
    "Python q / h": [math.floor(q / h) for q in queries],
    "jax.jit q / h": list(np.asarray(jax.jit(lambda a: jnp.floor(a / h))(jnp.asarray(queries)))),
    "yippy.sinc, jax.jit": list(np.asarray(
        jax.jit(lambda a: exact_quotient_floor_v1(a, 0.0, h))(jnp.asarray(queries)))),
}
fig, ax = plt.subplots(figsize=(8, 2.6), constrained_layout=True)
for y, (name, values) in enumerate(rows.items()):
    values = np.asarray(values, dtype=int)
    ax.scatter(offsets[values == 2], np.full((values == 2).sum(), y), color=pal.cyan, s=40)
    ax.scatter(offsets[values == 3], np.full((values == 3).sum(), y), color=pal.pink, s=40,
               marker="s")
ax.set_yticks(range(len(rows)), list(rows))
ax.invert_yaxis()
ax.set_xlabel("ULPs from the double nearest 3 x 0.38")
ax.set_ylim(3.5, -1.1)
ax.text(-6, -0.75, "cell 2", color=pal.cyan, ha="left", fontsize=9)
ax.text(6, -0.75, "cell 3", color=pal.pink, ha="right", fontsize=9)
plt.show()
```

Look for the column where each row switches from cyan to pink. Cyan dots are
queries assigned to cell 2, pink squares to cell 3. The yippy.sinc row matches the
exact row at every double, in eager and in compiled code.

## The remainder gives the exact floor

The yippy.sinc reader computes the floor from the remainder instead of the quotient. It
forms $d=s-o$ once, takes the C `fmod` remainder $r=\operatorname{fmod}(d,h)$,
which takes the sign of $d$ and is computed exactly, recovers the integer quotient as
$\operatorname{round}((d-r)/h)$ (a division whose exact result is an integer
far below $2^{53}$, so rounding cannot move it), and subtracts one when $d$
is negative with a nonzero remainder. The result is the floor of the exact
quotient of the two stored doubles $d$ and $h$, in every execution mode. The
supported range of spacings and origins, which keeps every intermediate a
normal number, is in {doc}`numerical-contract`.

```{code-cell} python
:tags: [hide-input]
import sys

from myst_nb import glue

rng = np.random.default_rng(11)
uniform = rng.uniform(-15, 15, 20000)
edge_queries = []
for k in range(-40, 41):  # cell edges k * h, which reach beyond 15
    low = high = k * h
    edge_queries.append(low)
    for _ in range(4):
        low, high = math.nextafter(low, -math.inf), math.nextafter(high, math.inf)
        edge_queries += [low, high]
test = np.concatenate([uniform, edge_queries])
# The contract flushes subnormal queries (the doubles around the edge at 0) to zero.
flushed = np.where(np.abs(test) < sys.float_info.min, 0.0, test)
exact_floor = np.array([math.floor(Fraction(float(x)) / Fraction(h)) for x in flushed])
x_test = jnp.asarray(test)
naive_jit = np.asarray(jax.jit(lambda a: jnp.floor(a / h))(x_test)).astype(int)
rule_eager = np.asarray(exact_quotient_floor_v1(x_test, 0.0, h)).astype(int)
rule_jit = np.asarray(jax.jit(lambda a: exact_quotient_floor_v1(a, 0.0, h))(x_test)).astype(int)
rule_vmap = np.asarray(jax.vmap(lambda a: exact_quotient_floor_v1(a, 0.0, h))(x_test)).astype(int)
glue("n_test", f"{test.size:,}", display=False)
glue("n_edges", f"{len(range(-40, 41))}", display=False)
glue("n_naive_wrong", f"{(naive_jit != exact_floor).sum()}", display=False)
glue("n_rule_wrong", f"{sum((r != exact_floor).sum() for r in (rule_eager, rule_jit, rule_vmap))}", display=False)
```

On a seeded test set of {glue:text}`n_test` queries (20,000 uniform on
$[-15,15]$ plus the doubles at and within four ULPs of {glue:text}`n_edges`
cell edges, at $h=0.38$), naive division under `jax.jit` chose a different
floor from exact arithmetic {glue:text}`n_naive_wrong` times, and the remainder
rule did so {glue:text}`n_rule_wrong` times in eager code, under `jax.jit` and
under `jax.vmap` combined. (The doubles around the edge at 0 are subnormal and
flush to zero by contract, so the exact floor is taken after the flush.) The count for the naive formula depends on the seed,
because the uniform queries rarely land near an edge, and the edge queries alone
fix the order of magnitude.

## A wrong floor costs a refusal, not accuracy

A floor that is off by one near a cell edge shifts the stencil by one node.
At an exact edge the node that leaves and the node that joins both have
weight exactly zero (a sinc evaluated at an integer), and a few ULPs away
their weights are tiny, so the image does not change measurably.

```{code-cell} python
:tags: [hide-input]
from yippy.sinc.kernel import stencil

end_weights = []
for q in queries:  # the doubles within six ULPs of the edge at 3 h
    _, w = stencil(jnp.asarray([q]), 0.0, h)
    end_weights.append(float(jnp.abs(w[0, [0, -1]]).max()))
glue("w_end", f"{max(end_weights):.0e}", display=False)
```

The cell above measures the weight of the first and last tap of the stencil for
every double within six ULPs of the edge at $3h$: the largest is
{glue:text}`w_end`, against a largest weight near 1.

The harm is
elsewhere: every node of the stencil must exist, so near the edge of the
stored grid a shifted stencil can reach one node too far and turn an
accepted read into a refusal, in one execution mode but not another. A
library whose decisions depend on how it was compiled cannot give
reproducible receipts, which is why the floor rule is versioned in the
kernel profile identity (`exact_quotient_floor_v1`).

Next: {doc}`performance`.
