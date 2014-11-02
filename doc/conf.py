# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from builder.shared_conf import *  # pyflakes:ignore (API import)

paths = ['../param/', '.', '..']
add_paths(paths)

# General information about the project.
project = u'Lancet'
copyright = u'2014, IOAM'
ioam_project = 'param'
from lancet import __version__

# The version info for the project being documented, defining |version|
# and |release| and used in various other places throughout the built
# documents.  Assumes __version__ is a param.version.Version object.
#
# The short X.Y.Z version.
version = __version__.abbrev()

# The full version, including alpha/beta/rc/dev tags.
release = __version__.abbrev(dev_suffix="-dev")

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'test_data', 'reference_data', 'nbpublisher',
                    'builder']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', 'builder/_shared_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'Lancetdoc'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/mini_logo.png'


# -- Options for LaTeX output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'Lancet.tex', u'Lancet Documentation',
   u'IOAM', 'manual'),
]


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'lancet', u'Lancet Documentation',
     [u'IOAM'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'Lancet', u'Lancet Documentation',
   u'IOAM', 'Lancet', 'One line description of project.',
   'Miscellaneous'),
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'http://docs.python.org/': None,
                       'http://ioam.github.io/param/': None,
                       'http://ioam.github.io/holoviews/': None,
                       'http://ipython.org/ipython-doc/2/' : None}


from builder.paramdoc import param_formatter

def setup(app):
    app.connect('autodoc-process-docstring', param_formatter)
