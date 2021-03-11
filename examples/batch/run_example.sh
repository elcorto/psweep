#!/bin/sh

set -eu

./clean.sh

cat > .gitignore << eof
calc/*/out.npy
calc/database_eval.pk
eof

./10input.py
cd calc
sh run_local.sh
cd ..
./20input.py
cd calc
sh run_local.sh
cd ..
./30eval.py

psweep-db2table calc/database_eval.pk param_a param_b mean _run_seq _pset_seq
