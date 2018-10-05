# publish on pypi
# ---------------
#   $ rm -rf dist
#   $ python3 setup.py sdist bdist_wheel
#   $ twine upload dist/*

import os, importlib
from setuptools import setup
from distutils.version import StrictVersion as Version

here = os.path.abspath(os.path.dirname(__file__))
bindir = 'bin'
with open(os.path.join(here, 'README.rst')) as fd:
    long_description = fd.read()

setup(
    name='psweep',
    version='0.2.1',
    description='loop like a pro, make parameter studies fun: set up and \
run a parameter study/sweep/scan, save a database',
    long_description=long_description,
    url='https://github.com/elcorto/psweep',
    author='Steve Schmerler',
    author_email='git@elcorto.com',
    license='BSD 3-Clause',
    keywords='parameter study sweep scan database pandas',
    packages=['psweep'],
    install_requires=open('requirements.txt').read().splitlines(),
    scripts=['{}/{}'.format(bindir, script) for script in os.listdir(bindir)]
)
