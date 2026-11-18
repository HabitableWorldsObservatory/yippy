"""Public catalog and loader for yield input packages (YIPs).

Each catalog entry describes a release asset and its download identity.
The FITS products inside the asset carry the optical metadata.

Public API:

- :func:`fetch_yip`: download and unpack a YIP by name or metadata filters.
- :func:`cache_dir`: locate the default YIP cache.
- :func:`list_yips`: list YIPs matching metadata filters.
- :func:`yip_exists`: check whether a YIP is in the catalog.
- :func:`yip_info`: retrieve a YIP's metadata.
- :func:`fetch_sinc_starter`: download and unpack the starter sinc delivery.
"""

from __future__ import annotations

import difflib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pooch
from pooch import Untar, Unzip

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


@dataclass(frozen=True)
class Archive:
    """A catalog YIP and the release asset that contains its native files."""

    name: str
    coronagraph: str
    designer: str
    sampling: str
    digest: str
    telescope: str | None = None
    release: str = "data-v2"

    @property
    def filename(self) -> str:
        """Name of the archive as published on the release."""
        return f"{self.name}.zip"

    @property
    def url(self) -> str:
        """Upstream URL of the release asset."""
        return (
            "https://github.com/HabitableWorldsObservatory/yippy/releases/download/"
            f"{self.release}/{self.filename}"
        )

    def archive_path(self, cache_path: str | Path | None = None) -> Path:
        """Locate the compressed archive without downloading or unpacking it."""
        root = cache_dir() if cache_path is None else Path(cache_path).expanduser()
        return root / self.filename


_CATALOG = {
    archive.name: archive
    for archive in (
        Archive(
            name="eac1_aavc_2d",
            telescope="eac1",
            coronagraph="aavc",
            designer="Susan Redmond",
            sampling="2d",
            digest="md5:1f4892faff18e55cbec9781a055bea4d",
        ),
        Archive(
            name="eac1_optimal_order_6_1d",
            telescope="eac1",
            coronagraph="optimal_order_6",
            designer="Rus Belikov",
            sampling="1d",
            digest="md5:df52540008a0e85467720ec91c3a84b8",
        ),
        Archive(
            name="usort_offaxis_ovc",
            coronagraph="offaxis_ovc",
            designer="Susan Redmond, Emiel Por",
            sampling="1d",
            digest="md5:f288b20f329412917d0a393a4d135439",
        ),
    )
}


# ---------------------------------------------------------------------------
# Starter sinc delivery
#
# One small public delivery (a physical-optics propagation of the public
# LUVOIR-A APLC design) that exercises the sinc-delivery-0.1 format. It is a
# separate tarball, not a YIP, so it is outside the YIP catalog.
# ---------------------------------------------------------------------------
SINC_STARTER_NAME: str = "sinc-starter"
SINC_STARTER_ARCHIVE: str = f"{SINC_STARTER_NAME}.tar.gz"
# The release asset is not published yet. Publishing it means replacing the
# placeholder tag below with the tag that carries the asset and setting
# SINC_STARTER_PUBLISHED to True.
SINC_STARTER_PUBLISHED: bool = False
SINC_STARTER_URL: str = (
    "https://github.com/HabitableWorldsObservatory/yippy/releases/download/"
    f"PLACEHOLDER-sinc-starter-release-tag/{SINC_STARTER_ARCHIVE}"
)
SINC_STARTER_SHA256: str = (
    "sha256:84c5c34cee908f89c1e6024afdc8661b735e43e1b0240c1fdd0888542b59c5af"
)


def _make_sinc_pikachu(cache_dir_path: str | Path) -> pooch.Pooch:
    """Build a pooch instance for the starter sinc delivery."""
    return pooch.create(
        path=cache_dir_path,
        base_url=SINC_STARTER_URL.rsplit("/", 1)[0] + "/",
        registry={SINC_STARTER_ARCHIVE: SINC_STARTER_SHA256},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_sinc_starter(cache_path: str | Path | None = None) -> str:
    """Download the starter sinc delivery (if not cached), unpack, return its path.

    The delivery holds 837 positive-quadrant samples, 16 interior checks and
    20 mirror checks on a 128 by 128 pixel frame, about 57 MB. It has no
    stellar or sky products, so check it with ``yippy-sinc validate``.

    Args:
        cache_path: Cache directory for this call. Defaults to :func:`cache_dir`.

    Returns:
        The path of the unpacked delivery directory.

    Raises:
        RuntimeError: The delivery is not published yet and no copy of the
            archive is in the cache directory.
    """
    root = Path(cache_path) if cache_path is not None else cache_dir()
    if not SINC_STARTER_PUBLISHED and not (root / SINC_STARTER_ARCHIVE).is_file():
        raise RuntimeError(
            "The starter sinc delivery is not published yet, so it cannot be "
            f"downloaded. To use a copy you already have, place "
            f"{SINC_STARTER_ARCHIVE} in {root}."
        )
    pikachu = _make_sinc_pikachu(root)
    paths = pikachu.fetch(SINC_STARTER_ARCHIVE, processor=Untar())
    return str(_yip_dir([Path(p) for p in paths], SINC_STARTER_NAME))


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
    return Path(pooch.os_cache("yippy"))


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

    if name is None:
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
        name = matches[0]

    archive = yip_info(name)
    path = archive.archive_path(cache_path)
    pikachu = pooch.create(
        path=path.parent,
        base_url="",
        urls={path.name: archive.url},
        registry={path.name: archive.digest},
    )
    logger.info(f"Fetching YIP {name!r} (cache: {path.parent})")
    paths = pikachu.fetch(path.name, processor=Unzip())
    yip_dir = _yip_dir([Path(p) for p in paths], name)
    logger.info(f"YIP {name!r} available at {yip_dir}")
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
    for name, archive in _CATALOG.items():
        if all(getattr(archive, k) == v for k, v in filters.items()):
            out.append(name)
    return out


def yip_exists(name: str) -> bool:
    """True iff ``name`` is an available YIP in the catalog."""
    return name in _CATALOG


def yip_info(name: str) -> Archive:
    """Describe a catalog asset without downloading it.

    Raises:
        KeyError: if ``name`` is not in the catalog.
    """
    try:
        return _CATALOG[name]
    except KeyError:
        suggestions = difflib.get_close_matches(name, _CATALOG, n=5)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise KeyError(f"Unknown YIP {name!r}.{hint}") from None
