#!/usr/bin/env python3

import random
from psweep import psweep as ps
import pandas as pd


def func(pset):
    return {'timing': random.random() * pset['a']}
            
                
if __name__ == '__main__':
    params = ps.seq2dicts('a', [1,2,3,4])
    df = pd.DataFrame()
    df = ps.run(df, func, params)
    ps.df_json_write(df, 'results.json')
