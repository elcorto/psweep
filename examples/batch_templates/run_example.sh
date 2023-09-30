#!/bin/sh

set -eux

./clean.sh

# Write files based on templates. See psweep.prep_batch().
./10input.py

# In this example, we execute the "batch jobs" locally (run_local.sh). In a
# real use case we would rsync (or psweep-push) calc/ to a remore machine, ssh,
# submit jobs there (run_cluster.sh), rsync (or psweep-pull) results back.
cd calc
sh run_local.sh

# Second run
cd ..
./15input.py
cd calc
sh run_local.sh

cd ..
./20eval.py

psweep-db2table calc/database_eval.pk param_a param_b mean _run_seq _pset_seq _run_id _pset_id
