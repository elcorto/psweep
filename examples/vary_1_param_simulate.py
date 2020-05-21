#!/usr/bin/env python3

import random
from itertools import product
import psweep as ps

def func(pset):
    return {'result': random.random() * pset['a'] * pset['b']}

if __name__ == '__main__':
    sel = ['_calc_dir', 'a', 'b', 'result']

    # 1st real run: produce some data, vary a, b constant
    params = ps.loops2params(product(
        ps.seq2dicts('a', [1,2,3,4]),
        ps.seq2dicts('b', [100])))
    df = ps.run(func, params)
    print(df[sel])

    # simulate run: check if new parameter grid is OK
    params = ps.loops2params(product(
        ps.seq2dicts('a', [5,6]),
        ps.seq2dicts('b', [88, 99])))
    df = ps.run(func, params, simulate=True)
    print(df[sel])

    # looks good, 2nd real run with new params
    df = ps.run(func, params)
    print(df[sel])
