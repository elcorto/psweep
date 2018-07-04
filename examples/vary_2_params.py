#!/usr/bin/env python3

#   _calc_dir                              _pset_id  \
# 0      calc  e5554177-ce31-4944-93ee-786dbaadacb7
# 1      calc  c055661c-fc36-476b-b50a-ae5c101ce638
# 2      calc  9f316933-5d46-42a7-aca8-14ccf3555ccc
# 3      calc  9aff8f32-402b-4f3a-9040-449a8a7e23c6
# 4      calc  0cf2a7f1-a5d6-4a23-8a9c-c14766b5d450
# 5      calc  268ba704-8c32-4bd8-8006-59bc3bd3c234
# 6      calc  c1732939-1668-4654-bb4f-8ad8c8391ef8
# 7      calc  9f79b241-0ef1-408c-a538-dc588a11a0de
# 
#                                 _run_id  a  b      result
# 0  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  1  8     5.95035
# 1  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  1  9     3.74252
# 2  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  2  8     2.58442
# 3  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  2  9  0.00564436
# 4  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  3  8     15.9873
# 5  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  3  9     19.2371
# 6  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  4  8     17.0561
# 7  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  4  9     21.0376


import random
from itertools import product
from psweep import psweep as ps


def func(pset):
    return {'result': random.random() * pset['a'] * pset['b']}
            
                
if __name__ == '__main__':
    params = ps.loops2params(product(
            ps.seq2dicts('a', [1,2,3,4]),
            ps.seq2dicts('b', [8,9]),
            ))
    df = ps.run(func, params)
    print(df)
