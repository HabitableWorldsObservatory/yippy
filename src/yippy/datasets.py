"""Public catalog and loader for yield input packages (YIPs).

The CATALOG dict is the source of truth: keys are flat YIP names like
``eac1_aavc_2d``; values carry just enough metadata to drive downloads
and filtered discovery (``telescope``, ``coronagraph``, ``sampling``,
``md5``). Descriptive metadata (designer, wavelengths, dark-zone
extent, ...) lives in the FITS headers inside each YIP, not here.

Archives are hosted as assets on a tagged GitHub release of this repo
(currently ``data-v1``) and fetched via pooch over HTTPS. The release
tag is separate from the code-release lifecycle managed by
release-please. To publish new YIPs: bump ``DATA_RELEASE_TAG`` to a new
``data-vN``, attach the updated zips to that release, and refresh the
md5 hashes here.

The catalog currently includes only the two reference YIPs used by the
yippy paper validation pipeline. Long-term YIP hosting will be provided
by ExEP, and when that catalog comes online the discovery API here will
grow back into a thin proxy over it.

Public API:
- ``fetch_yip(name=None, *, telescope=None, coronagraph=None, sampling=None,
              cache_path=None) -> str``
- ``cache_dir() -> Path``
- ``list_yips(**filters) -> list[str]``
- ``yip_exists(name) -> bool``
- ``yip_info(name) -> dict``
"""

from __future__ import annotations

import difflib
import logging
import os
from pathlib import Path
from typing import Any

import pooch
from pooch import Unzip

# By name, not relative import: scripts/build_zenodo_archives.py loads this
# module standalone via importlib and has no parent package.
logger = logging.getLogger("yippy")

# Quiet pooch's INFO-level chatter so YIP download events come through the
# yippy logger in the expected format. Warnings and errors still surface.
logging.getLogger("pooch").setLevel(logging.WARNING)

# Users can pin the YIP cache to a custom directory by exporting this env var.
# Resolution priority for a cache location:
#   1. ``cache_path`` keyword passed to fetch_yip
#   2. ``YIPPY_CACHE_DIR`` environment variable
#   3. pooch.os_cache("yippy") -- platform default via platformdirs
CACHE_DIR_ENV_VAR = "YIPPY_CACHE_DIR"

# ---------------------------------------------------------------------------
# Release tag carrying the YIP zip assets on this repo's GitHub releases.
# Bump to ``data-vN`` when the underlying YIP files change.
# ---------------------------------------------------------------------------
DATA_RELEASE_TAG: str = "data-v2"
_DATA_BASE_URL: str = f"https://github.com/HabitableWorldsObservatory/yippy/releases/download/{DATA_RELEASE_TAG}/"


# ---------------------------------------------------------------------------
# Catalog
#
# Currently a handful of reference YIPs from the Coronagraph Design Survey.
# Long-term YIP hosting will be provided by ExEP.
# ---------------------------------------------------------------------------
CATALOG: dict[str, dict[str, Any]] = {
    "eac1_aavc_2d": {
        "telescope": "eac1",
        "coronagraph": "aavc",
        "designer": "Susan Redmond",
        "md5": "md5:1f4892faff18e55cbec9781a055bea4d",
    },
    "eac1_optimal_order_6_1d": {
        "telescope": "eac1",
        "coronagraph": "optimal_order_6",
        "designer": "Rus Belikov",
        "md5": "md5:df52540008a0e85467720ec91c3a84b8",
    },
    "usort_offaxis_ovc": {
        # No telescope-architecture label; an off-axis vortex coronagraph
        # design study with no fixed EAC pairing.
        "coronagraph": "offaxis_ovc",
        "designer": "Susan Redmond, Emiel Por",
        "sampling": "1d",
        "md5": "md5:f288b20f329412917d0a393a4d135439",
    },
}


# Inject the sampling regime as a derived field on each catalog entry. Names
# that follow the ``{telescope}_{coronagraph}_(1d|2d)`` convention can omit
# ``sampling`` from their entry and have it parsed from the key suffix;
# entries whose name does not fit the convention (e.g. legacy or
# unconventionally-named YIPs) must set ``sampling`` explicitly.
for _name, _meta in CATALOG.items():
    if "sampling" in _meta:
        continue
    if _name.endswith("_1d"):
        _meta["sampling"] = "1d"
    elif _name.endswith("_2d"):
        _meta["sampling"] = "2d"
    else:
        raise RuntimeError(
            f"Catalog entry {_name!r} must either set 'sampling' explicitly "
            "or end with `_1d` / `_2d`."
        )
del _name, _meta


def _make_pikachu(cache_dir_path: str | Path) -> pooch.Pooch:
    """Build a pooch instance for the YIP catalog at ``cache_dir_path``."""
    return pooch.create(
        path=cache_dir_path,
        base_url=_DATA_BASE_URL,
        registry={f"{name}.zip": meta["md5"] for name, meta in CATALOG.items()},
    )


# Default pooch instance for yippy's YIP cache (platform-default location).
# Named after Corey's dog. The env-var and per-call overrides build their own
# pooch on demand; this is the fast path for zero-config users.
_PIKACHU = _make_pikachu(pooch.os_cache("yippy"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def cache_dir() -> Path:
    """Return the directory where yippy caches YIP archives by default.

    Resolution order:
      1. ``YIPPY_CACHE_DIR`` environment variable, if set.
      2. ``pooch.os_cache("yippy")`` -- the OS-conventional cache directory
         provided by platformdirs (e.g. ``~/Library/Caches/yippy`` on macOS,
         ``~/.cache/yippy`` on Linux).

    Override per call by passing ``cache_path`` to :func:`fetch_yip`.
    """
    env = os.environ.get(CACHE_DIR_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path(_PIKACHU.path)


def fetch_yip(
    name: str | None = None,
    *,
    telescope: str | None = None,
    coronagraph: str | None = None,
    sampling: str | None = None,
    cache_path: str | Path | None = None,
) -> str:
    """Download a YIP archive (if not cached), unpack, and return its path.

    Pass either ``name`` (flat: ``"eac1_aavc_2d"``) OR keyword filters
    (structured: ``telescope="eac1", coronagraph="aavc", sampling="2d"``).
    The keyword form must resolve to exactly one catalog entry; pass
    ``sampling`` whenever a ``(telescope, coronagraph)`` pair has both
    1D and 2D variants.

    YIPs are cached at :func:`cache_dir` (which honors the
    ``YIPPY_CACHE_DIR`` environment variable). Pass ``cache_path`` to
    override the cache location for this call only -- useful for
    shared institutional setups or project-scoped caches.

    Raises:
        TypeError: if both ``name`` and filters are passed (or neither).
        KeyError: if ``name`` is not in the catalog.
        ValueError: if the structured query has zero or multiple matches.
    """
    filters: dict[str, str] = {}
    if telescope is not None:
        filters["telescope"] = telescope
    if coronagraph is not None:
        filters["coronagraph"] = coronagraph
    if sampling is not None:
        filters["sampling"] = sampling

    if name is not None and filters:
        raise TypeError("Pass either `name` or filter kwargs, not both.")
    if name is None and not filters:
        raise TypeError("Pass either `name` or filter kwargs.")

    if name is not None:
        if name not in CATALOG:
            suggestions = difflib.get_close_matches(name, CATALOG.keys(), n=5)
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise KeyError(f"Unknown YIP {name!r}.{hint}")
        resolved = name
    else:
        matches = list_yips(**filters)
        if not matches:
            raise ValueError(
                f"No YIP matches {filters!r}. Try yippy.list_yips() to see options."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Filters {filters!r} matched multiple YIPs: {matches}. "
                "Pass `name=` directly or narrow filters."
            )
        resolved = matches[0]

    if cache_path is not None:
        pikachu = _make_pikachu(cache_path)
    elif os.environ.get(CACHE_DIR_ENV_VAR):
        pikachu = _make_pikachu(cache_dir())
    else:
        pikachu = _PIKACHU
    logger.info(f"Fetching YIP {resolved!r} (cache: {pikachu.path})")
    paths = pikachu.fetch(f"{resolved}.zip", processor=Unzip())
    yip_dir = _yip_dir([Path(p) for p in paths], resolved)

    logger.info(f"YIP {resolved!r} available at {yip_dir}")
    return str(yip_dir)


def _yip_dir(paths: list[Path], name: str) -> Path:
    """The YIP directory among the files Unzip reports for an archive.

    The YIP lives at the archive root under ``{name}/``. Unzip lists every
    file in the unzip folder, including ones the archive never held (a
    ``.DS_Store`` that Finder writes when the cache is browsed, say), so no
    single path is trusted: the first file under a folder named after the
    YIP decides. Without one, the first file's folder is walked up as before,
    treating a ``{name}.zip`` folder as the parent of the YIP folder.
    """
    for path in paths:
        for parent in path.parents:
            if parent.name == name:
                return parent
    sample_path = paths[0]
    yip_dir = sample_path.parent
    while yip_dir.parent != yip_dir:
        if yip_dir.name == f"{name}.zip":
            return yip_dir.parent / name
        yip_dir = yip_dir.parent
    return sample_path.parent


_FILTERABLE_FIELDS = frozenset({"telescope", "coronagraph", "sampling"})


def list_yips(**filters: str) -> list[str]:
    """Return catalog names matching all filters. No filters returns all names.

    Raises:
        TypeError: if a filter key is not a valid catalog field.
    """
    unknown = set(filters) - _FILTERABLE_FIELDS
    if unknown:
        raise TypeError(
            f"unknown filter keys: {sorted(unknown)}. "
            f"Valid keys are {sorted(_FILTERABLE_FIELDS)}."
        )
    out = []
    for name, meta in CATALOG.items():
        if all(meta.get(k) == v for k, v in filters.items()):
            out.append(name)
    return out


def yip_exists(name: str) -> bool:
    """True iff ``name`` is an available YIP in the catalog."""
    return name in CATALOG


def yip_info(name: str) -> dict[str, Any]:
    """Return the catalog metadata dict for ``name``.

    Raises:
        KeyError: if ``name`` is not in the catalog.
    """
    if name not in CATALOG:
        raise KeyError(name)
    return CATALOG[name]
