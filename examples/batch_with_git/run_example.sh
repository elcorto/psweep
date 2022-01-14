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
git commit -m "psweep manual: init test example"

# Write files based on templates.
#
#     templates
#     ├── calc
#     │   └── run.py
#     └── machines
#         ├── cluster
#         │   ├── info.yaml
#         │   └── jobscript
#         └── local
#             ├── info.yaml
#             └── jobscript
#
# We have machine templates for 2 machines, "local" and a remote machine named
# "cluster". psweep will generate run_<machine>.sh for both.
./10input.py

# In this example, we execute the "batch jobs" locally (run_local.sh). In a
# real use case we would rsync (or psweep-push) calc/ to a remore machine, ssh,
# submit jobs there (run_cluster.sh), rsync (or psweep-pull) results back.
cd calc
sh run_local.sh

# Modify driver script, prepare second run. For the sake of running this
# example automatically, we use an existing script and overwrite it to create
# some changes to the file we can track with git.
cd ..
mv _10input_run2.py 10input.py

git add -A
git commit -m "psweep manual: modify 10input.py"

./10input.py
cd calc
sh run_local.sh
cd ..
./20eval.py

psweep-db2table calc/database_eval.pk param_a param_b mean _run_seq _pset_seq
