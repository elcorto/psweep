import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
bindir = "bin"
with open(os.path.join(here, "README.md")) as fd:
    long_description = fd.read()

setup(
    name="psweep",
    version="0.5.0",
    description=(
        "loop like a pro, make parameter studies fun: set up and "
        "run a parameter study/sweep/scan, save a database"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elcorto/psweep",
    author="Steve Schmerler",
    author_email="git@elcorto.com",
    license="BSD 3-Clause",
    keywords="parameter study sweep scan database pandas",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=open("requirements.txt").read().splitlines(),
    scripts=["{}/{}".format(bindir, script) for script in os.listdir(bindir)],
)
