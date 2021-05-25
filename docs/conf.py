import sys
import os
import time
import numpoly

sys.path.insert(0, os.path.abspath('..'))

project = 'numpoly'
author = 'Jonathan Feinberg'
copyright = '%d, Jonathan Feinberg' % time.gmtime().tm_year
version = ".".join(numpoly.__version__.split(".")[:2])
master_doc = 'index'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

templates_path = ['.templates']
exclude_patterns = ['.build']

language = "en"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "tango"

# Create stubs automatically for all auto-summaries:
autosummary_generate = True
numpydoc_show_class_members = False

autodoc_type_aliases = {
    'ArrayLike': 'numpy.typing.ArrayLike',
    'DTypeLike': 'numpy.typing.DTypeLike',
    'PolyLike': 'numpoly.typing.PolyLike',
}

coverage_show_missing_items = True

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "collapse_navigation": True,
    "external_links":
        [{"name": "Github", "url": "https://github.com/jonathf/numpoly"}],
    "footer_items": ["sphinx-version.html"],
    "navbar_align": "left",
    "navbar_end": ["search-field.html"],
    "navigation_depth": 2,
    "show_prev_next": False,
}
html_short_title = "numpoly"
html_context = {"doc_path": "docs"}
html_logo = ".static/numpoly_logo2.svg"
html_static_path = ['.static']
html_sidebars = {"**": ["sidebar-nav-bs.html"]}

htmlhelp_basename = 'numpoly'
html_show_sourcelink = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
