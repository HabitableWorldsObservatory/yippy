---
file_format: mystnb
kernelspec:
  name: python3
---

# How do you check the files before sending them?

Once the images are propagated in one frame, in source-independent units, under
a symmetry argument that fits the design, one thing remains: confirming that
the files say what you mean, and sending them in a form the reader accepts. A
delivery that fails here fails after the campaign has run, so this page is
worth running on a small trial delivery before the full one.

:::{admonition} The question of this page
:class: important

What does a passing check prove, what does it not prove, and what goes in the
directory you send? Step 6 runs the validator and reads its summary, then lists
what to send. The close restates the three answers.
:::

```{code-cell} python
:tags: [remove-cell]
import csv
import sys

sys.path.insert(0, "../scripts")
import numpy as np
from IPython.display import Markdown
from myst_nb import glue
from validation_tables import load, static_dir


def tex(value, digits=2):
    """Format a number as inline math a x 10^b."""
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    return Markdown(rf"${mantissa}\times10^{{{int(exponent)}}}$")
```

(recipe-check)=
## Step 6: How do you check the files before sending?

Install yippy and run the validator on the delivery directory.
It runs the yippy integrity checks, then yippy.sinc admission with 64-bit
arithmetic, and it exits with 0 on a pass, 1 on a failure and 2 on a usage
error. The starter delivery is not on disk here, so the commands are shown and
not run.

```bash
pip install yippy
yippy-sinc validate path/to/delivery
yippy-sinc validate path/to/delivery --product mono-000 --json
```

The same check is available from Python. This example needs a delivery on
disk.

```python notest
from yippy.sinc.tools import validate

report = validate("path/to/delivery")
print(report.summary())
assert report.passed
```

The numbers a passing run reports on the starter come from the committed
validation data, which the cell reads. The first is the time yippy.sinc's
admission takes, the second the largest sampling margin on each axis, which
{doc}`designers` derived, and the rest are the worst and median interior check
and the largest axis residual.

```{code-cell} python
v = load("validation.json")
declared = load("floor_diagnosis.json")["band"]["declared_half_band"]
rho = 2 * np.array([declared["x"], declared["y"]]) * 0.38
rows = [r for r in csv.DictReader(open(static_dir() / "qualification_extended.csv"))
        if r["group"] == "starter"]
median = float(np.median([float(r["sinc_local_peak"]) for r in rows]))
worst = v["worst_check_error_local_peak"]
print(f"admission {v['admission_seconds']} s; rho {np.round(rho, 4)}")
print(f"interior worst {worst:.2e}, median {median:.2e}; axis residual {v['axis_residual_max']}")
print(f"level 1e-5 / worst = {1e-5 / worst:.1f}")
glue("adm_s", f"{v['admission_seconds']}", display=False)
glue("rho_x", f"{rho[0]:.4f}", display=False)
glue("rho_y", f"{rho[1]:.4f}", display=False)
glue("worst", tex(worst), display=False)
glue("median", tex(median), display=False)
glue("margin", f"{1e-5 / worst:.0f}", display=False)
```

On the starter, yippy.sinc's admission takes {glue:text}`adm_s` s and the run
passes with $\rho$ of {glue:text}`rho_x` and {glue:text}`rho_y`, all stored
axis images exactly invariant ({doc}`../validation` counts the 58), and an
interior worst and median of {glue}`worst` and {glue}`median` of the local
peak. The summary lines mean the following.

```{list-table}
:header-rows: 1
:widths: 22 40 38

* - Line
  - What it says
  - If it is wrong
* - `integrity`
  - hashes, headers, shapes and the complete rectangle agree with the manifest
  - fix the file or the manifest named in the message, then recompute the hash
* - `reasons`
  - the yippy.sinc refusal names, `none` when admitted
  - see the refusals below
* - `rho (x, y)`
  - occupancy per axis against the 0.76 limit
  - recompute the spacing from the shortest wavelength and the widest support
* - `interior checks`
  - worst and median residual of the read against direct propagation, as a
    fraction of the local peak
  - above $10^{-5}$ means a frame, normalization or convergence fault
* - `mirror checks`, `axis images`
  - negative-position images against the flipped stored images, and each
    stored axis image against its own flip, limit $10^{-6}$
  - a frame error or the wrong symmetry argument, see
    {ref}`step 5 <recipe-symmetry>`
```

The recommended interior level is $10^{-5}$ of the local peak, a round figure
about {glue:text}`margin` times above the {glue}`worst` that the starter's 16
checks read at worst. A tighter level of $10^{-6}$ would reject that clean
delivery, and a looser one would let through faults up to its own size. It is a recommendation from one clean delivery, not a bound
on reads between the checks. The mirror and axis limit of $10^{-6}$ is fixed by
the format.

The refusals that a first delivery meets, and the repair for each:

- `KERNEL_MARGIN_FAIL`: $\rho$ is above 0.76. The shortest wavelength is
  shorter than declared, or the support is wider than declared. Replan.
- `METADATA_MISSING`: a premise, the support evidence or the sampling
  wavelength is unset. Evidence code 2 is not accepted in version 0.1, so
  declare each premise with code 1 and a `source_id`
  ({doc}`declarations` lists the six premises).
- `SYMMETRY_CHECK_FAILED`: a mirror residual is above $10^{-6}$. Test the
  fiducial positions and the frame center, then reconsider the symmetry
  argument and fall back to the full grid.
- `GRID_INCOMPLETE` or a hole in the rectangle: a node is missing. Deliver the
  whole rectangle, including nodes inside the occulter.
- A missing local-peak reference: each interior check needs a positive
  `local_peak_reference`, the largest direct pixel within 1 $\lambda_0/D$ of
  the source.

### What does the validator not check?

Numerical convergence and storage precision are reported alongside the
validator output, and are not part of it. Rerun a handful of the interior
check positions at successive factor-two refinements of the propagation
sampling and the pixel integration, and report the change between levels as a
fraction of the finer image's local peak. Compare float32 storage with float64
at identical settings and positions, saving the float64 versions of the
checks. These are comparison levels to report, not gates, and a comparison
that was not run is reported as not run.

The validator does not run the fiducial test of {ref}`step 2 <recipe-frame>`,
and it cannot detect a consistent error, such as every image rotated by 180
degrees. Run that test yourself on the delivered cube.

## What do you send?

Send the directory the validator passed, unchanged: the manifest, the three
position lists, the training cube and offsets, the check and mirror-check
cubes, the mirror report and the refinement and storage comparisons the
request asks for. The layout and every key are on the [schema page][schema].
Two further items belong with the files.

- The stellar and sky products, in the legacy YIP layout, in the directory
  named by `legacy_package_path`. `open_yip(path).prepare(interpolation="sinc")`
  needs them, so a delivery meant for direct use must include them. The
  starter holds none, and that is why it is read through the router.
- The validator summary, any failed checks reported as failed, and the
  fiducial-test centroids.

## What this page established

A passing validator proves that the files agree with their manifest and with
each other, at the level of the starter's clean run, and says nothing about a
consistent error such as a wrong frame. The fiducial test and the convergence
and storage comparisons are yours to run and report, and what you send is the
validated directory, unchanged, with the stellar and sky products and the
summary. The last page of the guide, {doc}`../validation`, shows the measured
evidence for the whole chain.

Next: {doc}`../validation`.

[schema]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html
