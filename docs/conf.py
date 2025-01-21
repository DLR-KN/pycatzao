# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import re

from pycatzao import __version__  # noqa: E402

project = "Pycatzao"

# TODO: add copyright
copyright = ""

author = "Nis Meinert"

# The short X.Y version
parsed_version = re.match(r"(\d+\.\d+)", __version__)
version = parsed_version.group(1) if parsed_version else __version__

# The full version, including alpha/beta/rc tags
parsed_release = re.match(r"(\d+\.\d+\.\d+)", __version__)
release = parsed_release.group(1) if parsed_release else __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
}

templates_path = []
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "classic"
html_static_path = ["static"]
