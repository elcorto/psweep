#!/usr/bin/env python3

# run 10multiple*.py first.

import random
from itertools import product
from psweep import psweep as ps


def func(pset):
    return {'result': random.random() * pset['a'] * pset['b']}


if __name__ == '__main__':
    
    print("second run")

    const = {'a': 11111,
             'b': 55555}

    params = []
    disp_cols = []
    
    # second run: extend study with more 'b' values
    study = 'b'
    vary = ps.seq2dicts('b', [88,99])
    this_params = ps.loops2params(product(vary, [{'study': study}]))
    this_params = [ps.merge_dicts(const, dct) for dct in this_params]
    params += this_params

    df = ps.run(func, params, backup_script=__file__, backup_calc_dir=True)
    print(df[['a', 'b', 'result', '_run_id']])
