# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Tinygrad'
copyright = '2023, Tiny Corp'
author = 'Tiny Corp'
release = '0.7.0'

# -- Tinygrad Path Setup -----------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
        # Documentation from docstrings
    "sphinx.ext.napoleon", 
        # NOTE: "Enables NumPy or Google style docstrings over reStructuredText. 
        # Napoleon converts the docstrings to correct reStructuredText before autodoc processes them
    "sphinx.ext.extlinks",
        # Markup to shorten external links
    "sphinx.ext.intersphinx",
        # Link to other project's documentation
    "sphinx.ext.mathjax",
        # Display math in HTML via JavaScript
    "sphinx.ext.todo",
        # Support for todo items
    "sphinx.ext.viewcode", 
        # Add links to highlighted source code,
    "nbsphinx",
        # Jupyter Notebook support
    "sphinx_copybutton", 
        # A small sphinx extension to add a "copy" button to code blocks 
    "sphinx_inline_tabs", 
        # Inline tabbed content for Sphinx
    "myst_parser"
        # Markdown parser for Sphinx
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for intersphinx -------------------------------------------------
#

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}

pygments_style = "emacs"
# pygments_dark_style = "monokai"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_logo = './_static/logo.png' # Sphinx mechanism to add your project's logo
html_static_path = ['_static']
html_css_files = [
      "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/fontawesome.min.css",
      "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/solid.min.css",
      "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/brands.min.css",
]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "announcement": "The API is still a big work in progress, but we're getting there!",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/geohot/tinygrad",
            "html": "",
            "class": "fa-brands fa-solid fa-github fa-lg",
        },
        {
            "name": "X",
            "url": "https://x.com/__tinygrad__",
            "html": "",
            "class": "fa-brands fa-solid fa-x-twitter fa-lg",
        },
        {
            "name": "Discord",
            "url": "https://discord.gg/ZjZadyC7PK",
            "html": "",
            "class": "fa-brands fa-solid fa-discord fa-lg",
        },
        {
            "name": "Website",
            "url": "https://tinygrad.org",
            "html": "",
            "class": "fa-solid fa-globe fa-lg",
        },
    ]
}