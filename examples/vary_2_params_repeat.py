#!/usr/bin/env python3

import random
from itertools import product
import psweep as ps


def func(pset):
    return {'result': random.random() * pset['a'] * pset['b']}


if __name__ == '__main__':
    params = ps.loops2params(product(
            ps.seq2dicts('a', [1,2,3,4]),
            ps.seq2dicts('b', [8,9]),
            ))
    # First run.
    df = ps.run(func, params)

    # Second run. 
    params = ps.loops2params(product(
            ps.seq2dicts('a', [11,22,33,44]),
            ps.seq2dicts('b', [88,99]),
            ))
    df = ps.run(func, params)
    print(df)
