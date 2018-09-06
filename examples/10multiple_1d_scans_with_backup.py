#!/usr/bin/env python3

# 1d scans of either parameter a or b, using constant defaults for the other
# from the `const` dict.
#
# $ rm -rf calc*; ./10multiple*.py; ./20multiple*.py 

import random
from itertools import product
from psweep import psweep as ps


def func(pset):
    return {'result': random.random() * pset['a'] * pset['b']}


if __name__ == '__main__':

    print("first run")

    const = {'a': 11111,
             'b': 55555}

    params = []
    disp_cols = []
    
    # vary only a
    study = 'a'
    vary = ps.seq2dicts('a', [1,2,3])
    this_params = ps.loops2params(product(vary, [{'study': study}]))
    this_params = [ps.merge_dicts(const, dct) for dct in this_params]
    params += this_params
    
    # vary only b
    study = 'b'
    vary = ps.seq2dicts('b', [66,77])
    this_params = ps.loops2params(product(vary, [{'study': study}]))
    this_params = [ps.merge_dicts(const, dct) for dct in this_params]
    params += this_params
    
    df = ps.run(func, params, backup_script=__file__, backup_calc_dir=True)
    print(df[['a', 'b', 'result', '_run_id']])
