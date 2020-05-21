#!/usr/bin/env python3

import random
import psweep as ps

def func(pset):
    return {'result': random.random() * pset['a']}
                
if __name__ == '__main__':
    params = ps.seq2dicts('a', [1,2,3,4])
    df = ps.run(func, params)
    print(df)
