#!/usr/bin/env python3

"""
First run. Create

  param_a    param_b    _run_seq    _pset_seq  _run_id
---------  ---------  ----------  -----------  ------------------------------------
        1         88           0            0  c9300a55-dfd7-431c-9e06-526328807417
        1         99           0            1  c9300a55-dfd7-431c-9e06-526328807417
        2         88           0            2  c9300a55-dfd7-431c-9e06-526328807417
        2         99           0            3  c9300a55-dfd7-431c-9e06-526328807417
        3         88           0            4  c9300a55-dfd7-431c-9e06-526328807417
        3         99           0            5  c9300a55-dfd7-431c-9e06-526328807417

Note: _run_id is a random UUID which will be different every time you run this
script.
"""

import psweep as ps


if __name__ == "__main__":
    param_a = ps.plist("param_a", [1, 2, 3])
    param_b = ps.plist("param_b", [88, 99])

    params = ps.pgrid(param_a, param_b)
    df = ps.prep_batch(params, git=True, template_mode="dollar")
    print(df)
