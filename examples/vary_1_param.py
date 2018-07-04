#!/usr/bin/env python3

#   _calc_dir                              _pset_id  \
# 0      calc  736d6ebf-12a3-488b-bb14-f78a1d2324d8
# 1      calc  5a0e88a5-07ba-43d6-b4b3-28f85330c293
# 2      calc  9511fa11-1017-4cb0-a298-29cb8e86095e
# 3      calc  7b9abdac-35fc-4f57-a7ff-339f39a7c03d
# 
#                                 _run_id  a    result
# 0  bce7f771-39c2-4487-afbc-3fd484bb5a6a  1  0.353634
# 1  bce7f771-39c2-4487-afbc-3fd484bb5a6a  2    1.1975
# 2  bce7f771-39c2-4487-afbc-3fd484bb5a6a  3   1.95837
# 3  bce7f771-39c2-4487-afbc-3fd484bb5a6a  4   1.10552

import random
from psweep import psweep as ps

def func(pset):
    return {'result': random.random() * pset['a']}
                
if __name__ == '__main__':
    params = ps.seq2dicts('a', [1,2,3,4])
    df = ps.run(func, params)
    print(df)
