# publish on pypi
# ---------------
#   $ python3 setup.py sdist
#   $ twine upload dist/psweep-x.y.z.tar.gz 

import os, importlib
from setuptools import setup
from distutils.version import StrictVersion as Version

here = os.path.abspath(os.path.dirname(__file__))
bindir = 'bin'
with open(os.path.join(here, 'README.rst')) as fd:
    long_description = fd.read()

# Hack to make pip respect system packages. So sad! Why in the world can pip
# list and uninstall system packages (Debian: /usr/lib/python3/dist-packages),
# but NOT respect them when installing!!!?? It completely ignores existing
# packages, even with correct versions, and happily re-installs them from
# it's own sources (of course to other locations). Then we end up with two
# installations of the SAME package in the SAME version -- thank you very much.
# I don't get it, seriously.
install_requires = []

req = [('pandas', 'pandas', '>=', '0.19.2')]

for pipname,pkgname,op,ver in req:
    print("checking dependency: {}".format(pkgname))
    req = pipname + op + ver if op and ver else pipname
    try:
        pkg = importlib.import_module(pkgname)
        if op and ver:
            cmd = "Version(pkg.__version__) {op} Version('{ver}')".format(op=op,
                                                                          ver=ver)
            if not eval(cmd):
                install_requires.append(req)
    except ImportError:
        install_requires.append(req)

print("install_requires: {}".format(install_requires))


setup(
    name='psweep',
    version='0.1.0',
    description='loop like a pro, make parameter studies fun',
    long_description=long_description,
    url='https://github.com/elcorto/psweep',
    author='Steve Schmerler',
    author_email='git@elcorto.com',
    license='GPLv3',
    keywords='parameter study sweep loop',
    packages=['psweep'],
    install_requires=install_requires,
    scripts=['{}/{}'.format(bindir, script) for script in os.listdir(bindir)]
)
