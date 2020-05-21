#!/usr/bin/env python3

"""
1d scans of either parameter a or b, using constant defaults for the other
from the `const` dict. We use merge_dicts() to combine constant and varied
values. Note that merge_dicts() operates left-to-right, so const must be
left, i.e. replace const values which are varied.

We also use the backup_* options to show that they can be always used (also
when there is no data yet).

run:
    $ rm -rf calc*; ./10multiple*.py; ./20multiple*.py
"""

import random
import psweep as ps


def func(pset):
    return {'result': random.random() * pset['a'] * pset['b']}


if __name__ == '__main__':

    print("first run")

    const = {'a': 11111,
             'b': 55555}

    params = []
    disp_cols = []

    values = dict(a=[1,2,3], b=[66,77])

    for study,seq in values.items():
        # [{'a': 1}, {'a': 2}, {'a': 3}]
        # [{'b': 66}, {'b': 77}]
        params_1d = ps.plist(study, seq)
        this_params = ps.pgrid(params_1d, [{'study': study}])
        this_params = [ps.merge_dicts(const, dct) for dct in this_params]
        params += this_params
        disp_cols.append(study)

    disp_cols += ['_run_id']
    df = ps.run(func, params, backup_script=__file__, backup_calc_dir=True,
                verbose=disp_cols)
    print(df[disp_cols + ['result']])
