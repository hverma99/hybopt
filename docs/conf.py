"""Sphinx configuration for the hybopt documentation."""

from importlib.metadata import version as _version

project = "hybopt"
author = "Brooke Soobrian, Harshit Verma, Zhe Li, Kouroche Bouchiat"
release = _version("hybopt")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
]

# Figures drawn by ``.. plot::`` in docstrings (e.g. hybopt.methods.tanh), from the package code.
plot_include_source = False
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 150)]
plot_basedir = "figures"  # scripts named in ``.. plot:: <file>.py``

# Generate a page per module for every autosummary :recursive: listing.
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pyomo": ("https://pyomo.readthedocs.io/en/stable", None),
}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
# Only the Sphinx sources belong in the build; the rest of docs/ is project paperwork.
exclude_patterns = ["_build", "description.md", "proposal.tex"]

html_theme = "furo"
html_title = "hybopt"
