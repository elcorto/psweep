#!/usr/bin/env python3

import random
from itertools import product
import psweep as ps

def func(pset):
    return {'result': random.random() * pset['a']}

if __name__ == '__main__':
    params = ps.loops2params(product(ps.seq2dicts('a', [1,2,3,4]),
                                     [{'study': 'a'}]))
    df = ps.run(func, params)
    print(df)
