project = "psweep"
author = "Steve Schmerler"
copyright = "2026, Steve Schmerler"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

root_doc = "index"
templates_path = []
exclude_patterns = []

add_module_names = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "inherited-members": True,
    "no-special-members": True,
}

html_theme = "sphinx_book_theme"
html_logo = "psweep-logo.png"
html_static_path = []
html_theme_options = {
    "repository_url": "https://github.com/elcorto/psweep",
    "repository_branch": "main",
    "path_to_docs": "doc/source",
    "show_toc_level": 2,
    "use_issues_button": False,
    "use_repository_button": True,
}
