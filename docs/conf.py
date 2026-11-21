"""Sphinx configuration file."""

import os
from pathlib import Path

import yippy

# Executed sinc pages import the toy coronagraph helper that lives under docs/sinc;
# notebook kernels inherit the environment of the build process.
_SINC_DOCS = str(Path(__file__).resolve().parent / "sinc")
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (_SINC_DOCS, os.environ.get("PYTHONPATH", "")) if p
)

project = "yippy"
copyright = "2024, Corey Spohn"
author = "Corey Spohn"
version = yippy.__version__
release = yippy.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxcontrib.mermaid",
    "sphinx_design",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "skyscapes": ("https://skyscapes.readthedocs.io/en/latest/", None),
    "optixstuff": ("https://optixstuff.readthedocs.io/en/latest/", None),
    "coronagraphoto": ("https://coronagraphoto.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", ".build", "Thumbs.db", ".DS_Store", "sinc/scripts"]

language = "en"

autoapi_dirs = ["../src"]
autodoc_typehints = "description"
# AutoAPI already indexes fields; Napoleon should render their descriptions only.
napoleon_use_ivar = True

myst_enable_extensions = ["amsmath", "dollarmath", "colon_fence", "deflist"]
myst_fence_as_directive = ["mermaid"]
# The sinc pages carry generated values, so notebooks without outputs execute.
nb_output_stderr = "remove"
nb_merge_streams = True

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["sinc-outputs.css"]
master_doc = "index"
html_favicon = "_static/favicon.ico"
html_title = "Yield Input Package Python Wrapper"
html_sidebars = {"posts/*": ["sbt-sidebar-nav.html"]}

html_theme_options = {
    "repository_url": "https://github.com/HabitableWorldsObservatory/yippy",
    "repository_branch": "main",
    "use_repository_button": True,
    "show_toc_level": 2,
}
html_context = {
    "default_mode": "dark",
}
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}
nb_execution_mode = "auto"
nb_execution_timeout = 300
# These tutorials require user-supplied YIP directories under input/.
nb_execution_excludepatterns = [
    "examples/Basics.ipynb",
    "examples/QuarterSymmetric.ipynb",
    "examples/RadiallySymmetric.ipynb",
    "examples/SkyTrans.ipynb",
]


def _generate_yip_catalog_table(app):
    """Pre-build hook: write docs/_generated/yip_catalog.md from the archive catalog.

    Keeps the public docs page in lockstep with the catalog source.
    """
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    from yippy.datasets import list_yips, yip_info

    rows = [
        "| YIP name | Telescope | Coronagraph | Sampling | Designer |",
        "|---|---|---|---|---|",
    ]
    # Telescope is optional for design-study YIPs that have no fixed
    # telescope-architecture pairing; render those cells as an em dash.
    for name in sorted(list_yips()):
        archive = yip_info(name)
        telescope = archive.telescope
        telescope_cell = f"`{telescope}`" if telescope else "--"
        rows.append(
            f"| `{name}` | {telescope_cell} | `{archive.coronagraph}` | "
            f"`{archive.sampling}` | {archive.designer} |"
        )

    out_dir = Path(__file__).parent / "_generated"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "yip_catalog.md").write_text("\n".join(rows) + "\n")


def _skip_autoapi_member(app, what, name, obj, skip, options):
    """Keep generated version details and the colliding logger alias out of the API."""
    # Parse the generated file so its public __version__ import resolves, but
    # document only the package-level version and the logger module.
    if name == "yippy._version" or name.startswith("yippy._version."):
        return True
    if what == "data" and name == "yippy.logger":
        return True
    return skip


def setup(app):
    """Register Sphinx hooks for yippy's docs build."""
    app.connect("builder-inited", _generate_yip_catalog_table)
    app.connect("autoapi-skip-member", _skip_autoapi_member)
