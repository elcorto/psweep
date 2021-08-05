#!/bin/sh

# git status exits 0 if run in a git repo and != 0 else.
if git status > /dev/null 2>&1; then
    cat << eof
We seem to run in a git repo, abort. This script must not be run inside
psweep's repo, because it will create commits!

To run this example, copy the dir to some temp location, which we also do in
the tests.
eof
    exit 1
fi

set -eux

./clean.sh

cat > .gitignore << eof
calc/*/out.npy
calc/database_eval.pk
eof

git init
git add -A
git commit -m "psweep: init test example"

./10input.py
cd calc
sh run_local.sh
cd ..
mv 20input.py 10input.py

git add -A
git commit -m "psweep: modify 10input.py"

./10input.py
cd calc
sh run_local.sh
cd ..
./30eval.py

psweep-db2table calc/database_eval.pk param_a param_b mean _run_seq _pset_seq
