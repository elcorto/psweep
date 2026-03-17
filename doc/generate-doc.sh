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

apidoc=sphinx-apidoc
command -v "$apidoc" > /dev/null 2>&1 || err "executable $apidoc not found"

# Closer to our previous generated API docs than the sphinx-apidoc defaults.
apidoc_member_opts="members,show-inheritance,inherited-members"

# Ensure a clean generated tree.
rm -rf $(find $package_dir -name "*.pyc" -o -name "__pycache__")
rm -rf build/ source/_build source/generated/

# Generate API doc rst files.
SPHINX_APIDOC_OPTIONS=$apidoc_member_opts \
    $apidoc \
    -f \
    --remove-old \
    -e \
    --doc-project "API Reference" \
    --module-first \
    --tocfile index \
    -o source/generated/api \
    ../src/$package_name

sphinx-build -M html source build
