#!/bin/sh

set -eu

err(){
    # shellcheck disable=SC2145
    echo "error: $@"
    exit 1
}

# We assume this Sphinx layout.
#
#   /path/to/package_name
#   ├── doc                     <-- here
#   │   ├── generate-doc.sh
#   │   └── source
#   │       ├── conf.py
#   │       ├── index.md
#   │       ├── some_docs.md
#   │       ...
#   ├── setup.py
#   ├── src/package_name/
#   ...

# /path/to/package_name
package_dir=$(readlink -f ../)

# package_name
package_name=$(basename $package_dir)

##autodoc_extra_opts="--write-doc"
autodoc_extra_opts=

autodoc=sphinx-autodoc
command -v "$autodoc" > /dev/null 2>&1 || err "executable $autodoc not found"

# Ensure a clean generated tree.
rm -rf $(find $package_dir -name "*.pyc" -o -name "__pycache__")
rm -rf build/ source/_build source/generated/ source/_autosummary

# generate API doc rst files
$autodoc $autodoc_extra_opts -s source -a generated/api \
    -X 'test[s]*\.test_' $package_name

sphinx-build -M html source build
