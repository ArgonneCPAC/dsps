"""
"""
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("emcee").version
except DistributionNotFound:
    __version__ = "unknown version"


project = "dsps"
copyright = "2023, Andrew Hearin"
author = "Andrew Hearin"
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = [".ipynb_checkpoints/*"]

nbsphinx_kernel_name = "python3"

add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
