#!/usr/bin/env python3

"""
After

  $ cd calc
  $ sh run_local.sh
  $ cd ..

create more psets (_run_seq=1)

  param_a    param_b    _run_seq    _pset_seq  _run_id
---------  ---------  ----------  -----------  ------------------------------------
        1         88           0            0  c9300a55-dfd7-431c-9e06-526328807417  # old
        1         99           0            1  c9300a55-dfd7-431c-9e06-526328807417  #
        2         88           0            2  c9300a55-dfd7-431c-9e06-526328807417  #
        2         99           0            3  c9300a55-dfd7-431c-9e06-526328807417  #
        3         88           0            4  c9300a55-dfd7-431c-9e06-526328807417  #
        3         99           0            5  c9300a55-dfd7-431c-9e06-526328807417  #

        5        100           1            6  9f953b9a-4da0-4c67-9fcd-6e18d0b8f36a
        6        100           1            7  9f953b9a-4da0-4c67-9fcd-6e18d0b8f36a

NOTE: In a normal workflow, where we iteratively extend the study, we would
instead change 10input.py and track changes by git instead of creating a new
*input.py script like this one here. We do this here only for the sake of
automatically running the example!
"""

import psweep as ps

if __name__ == "__main__":
    param_a = ps.plist("param_a", [5, 6])
    param_b = ps.plist("param_b", [100])

    params = ps.pgrid(param_a, param_b)
    df = ps.prep_batch(params, git=True)
    print(df)
