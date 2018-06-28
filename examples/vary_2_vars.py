#!/usr/bin/env python3

import random
from itertools import product
from psweep import psweep as ps
import pandas as pd


def func(pset):
    return {'timing': random.random() * pset['a'] * pset['b']}
            
                
if __name__ == '__main__':
    params = ps.loops2params(product(
            ps.seq2dicts('a', [1,2,3,4]),
            ps.seq2dicts('b', [1.0,2.0]),
            ))
    df = pd.DataFrame()
    df = ps.run(df, func, params)
    ps.df_json_write(df, 'results.json')
