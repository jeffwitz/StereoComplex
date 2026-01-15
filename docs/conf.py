from __future__ import annotations

import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)


project = "StereoComplex"
author = "StereoComplex contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "assets/rayfield_worked_example/_out/**",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Number figures/tables/code-blocks so docs can reference them like a paper.
numfig = True
numfig_secnum_depth = 2
numfig_format = {
    "figure": "Fig. %s",
    "table": "Tab. %s",
    "code-block": "Listing %s",
}

# PDF build: use XeLaTeX for robust UTF-8 handling.
latex_engine = "xelatex"
