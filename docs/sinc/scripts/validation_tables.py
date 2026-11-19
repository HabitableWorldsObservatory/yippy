"""Format the committed validation JSON files into the tables the pages show.

Every number on ``validation.md`` and ``reference/performance.md`` is read from
``_static/validation/*.json`` by these helpers, so a page cannot drift from the
files the scripts wrote.
"""

import json
from pathlib import Path


def static_dir():
    """Locate ``_static/validation`` from the page's working directory."""
    for base in (Path("."), Path("docs"), Path(__file__).resolve().parent.parent):
        if (base / "_static" / "validation").is_dir():
            return base / "_static" / "validation"
    raise FileNotFoundError("_static/validation")


def load(name):
    """Read one JSON file written by a script under ``docs/scripts``."""
    return json.loads((static_dir() / name).read_text())


def sci(value, digits=2):
    """Format a number as inline math ``a x 10^b``; zero stays 0."""
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


BIN_LABELS = {
    "behind the inner working angle": "Just inside the design inner working angle",
    "inner working angle to +0.2": "From the design angle to 0.2 above it",
    "beyond the inner working angle": "Beyond that, up to the largest radius read",
}


def bin_tables(result, method, radii=None):
    """Markdown tables, one per radius bin, of the three metrics for ``method``.

    ``radii`` lists the source radius of every read. When given, each heading
    states the smallest and largest radius actually present in the bin, and
    not the bin edges.
    """
    names = {
        "local_peak": "error over the local peak",
        "pixel_brightest": "error over each pixel's brightest value",
        "photometric": "core flux error (aperture radius 0.7)",
    }
    out = []
    for entry in result["bins"]:
        if radii is None:
            upper = "" if entry["r_max"] is None else f" to {entry['r_max']:.2f}"
            span = f"{entry['r_min']:.2f}{upper}"
        else:
            inside = [
                r
                for r in radii
                if r >= entry["r_min"]
                and (entry["r_max"] is None or r < entry["r_max"])
            ]
            span = f"{min(inside):.2f} to {max(inside):.2f}"
        out.append(
            f"**{BIN_LABELS.get(entry['name'], entry['name'])}** (source radii {span} "
            f"$\\lambda_0/D$, {entry['n']} positions)\n"
        )
        out.append("| Metric | Median | 95th percentile | Worst |")
        out.append("|---|---|---|---|")
        for key, label in names.items():
            row = entry[method][key]
            cells = [sci(row[k]) for k in ("median", "p95", "worst")]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)
