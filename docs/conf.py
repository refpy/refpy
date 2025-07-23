import os
import sys

sys.path.insert(0, os.path.abspath("../"))

project = 'RefPy'
copyright = '2025, Ismael Ripoll'
author = 'Ismael Ripoll'
release = 'v0.1.12'

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autosummary',
              'sphinx.ext.autodoc', 'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
autosummary_generate = True  # Automatically generate summary tables and stub files

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs"]  # Show the sidebar and search field everywhere
}

html_show_sourcelink = False

autodoc_member_order = 'bysource'

html_theme_options = {
    "show_nav_level": 2,               # Show second-level navigation
    "collapse_navigation": False,      # Collapses the submenus in the sidebar
    "navigation_depth": 4,             # Sidebar navigation depth
    "navigation_with_keys": True,
    "icon_links": [                    # Icons in the top bar (e.g., GitHub)
        {
            "name": "GitHub",
            "url": "https://github.com/refpy/refpy/",
            "icon": "fab fa-github",
        },
    ],
    "footer_start": ["copyright"],      # Footer customization
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo_dark.svg",
    }
}