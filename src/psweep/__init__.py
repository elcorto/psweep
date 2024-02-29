import importlib.metadata
from .psweep import *

# https://stackoverflow.com/a/67097076
#
# The version will be the same as the one in pyproject.toml
#
#   [project]
#   name = "psweep"
#   version = "X.Y.Z"
#
# We could also do it the other way around and define the version string here
# and then with setuptools at least
# (https://packaging.python.org/en/latest/guides/single-sourcing-package-version/#single-sourcing-the-version)
#
#   [project]
#   name = "psweep"
#   dynamic = ["version"]
#
#   [tool.setuptools.dynamic]
#   version = {attr = "package.__version__"}
#
# But we don't, so we are able to switch out the build backend, if the
# need arises.
#
__version__ = importlib.metadata.version(__package__)
