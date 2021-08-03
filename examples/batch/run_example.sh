#!/bin/sh

set -eux


./clean.sh

# Buy writing this, we create a dirty repo, which will make the first
# ps.run_local() call fail since by default we (hopefully :)) don't commit
# local changes. This will prevent people from accidentally committing to a
# psweep checkout.
#
# To run this example, copy the dir to some temp location, which we also do in
# the tests.
cat > .gitignore << eof
calc/*/out.npy
calc/database_eval.pk
eof

./10input.py
cd calc
sh run_local.sh
cd ..
cp 10input.py 10input.py.bak
cp 20input.py 10input.py
./10input.py
cd calc
sh run_local.sh
cd ..
./30eval.py

psweep-db2table calc/database_eval.pk param_a param_b mean _run_seq _pset_seq

mv 10input.py.bak 10input.py
