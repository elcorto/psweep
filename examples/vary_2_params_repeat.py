#!/usr/bin/env python3

#    _calc_dir                              _pset_id  \
# 0       calc  cdaba219-adcd-4b36-bb27-5c9aea922396
# 1       calc  196acfdb-f494-4be7-a927-ae8994d32061
# 2       calc  168ae386-16b4-4c00-8c4e-0cfc719b3453
# 3       calc  08d8e5c2-4efe-444a-8945-eb3679bda540
# 4       calc  8958936e-98ee-4ef4-b867-24e64fa39065
# 5       calc  b9f5bd76-4c6d-47f4-9bdb-82e193aadebd
# 6       calc  e175cfb1-bcd6-4a67-a268-ed8deb71dab8
# 7       calc  17154ba9-3e25-4f94-a0ed-c335d304e6db
# 8       calc  93564543-a80e-4ba8-b83c-b5f228d84a99
# 9       calc  395e6ddc-c9f1-4920-aebd-5a5478997356
# 10      calc  de6dfdc7-1c7c-465f-84ad-a073c7534dba
# 11      calc  a07e1865-a628-4b64-a09c-248de042969d
# 12      calc  cf71c40d-4816-4607-80d3-5b59ac6e3a67
# 13      calc  cb903525-17d5-486e-b297-7e484ebac380
# 14      calc  400d2bf5-b9e0-4446-a46c-b302f3fa060c
# 15      calc  c27d0f85-3a16-4184-b522-7a62e8066d8a
# 
#                                  _run_id   a   b   result
# 0   3b429565-a2be-478b-a9f3-4490a1af9bfb   1   8   1.5881
# 1   3b429565-a2be-478b-a9f3-4490a1af9bfb   1   9  6.27259
# 2   3b429565-a2be-478b-a9f3-4490a1af9bfb   2   8  7.16291
# 3   3b429565-a2be-478b-a9f3-4490a1af9bfb   2   9  13.9371
# 4   3b429565-a2be-478b-a9f3-4490a1af9bfb   3   8  11.4468
# 5   3b429565-a2be-478b-a9f3-4490a1af9bfb   3   9  7.65542
# 6   3b429565-a2be-478b-a9f3-4490a1af9bfb   4   8  21.8727
# 7   3b429565-a2be-478b-a9f3-4490a1af9bfb   4   9  19.7677
# 8   f86b2f28-58b1-4459-93be-1396b3f8c76d  11  88  103.765
# 9   f86b2f28-58b1-4459-93be-1396b3f8c76d  11  99   282.79
# 10  f86b2f28-58b1-4459-93be-1396b3f8c76d  22  88   1680.6
# 11  f86b2f28-58b1-4459-93be-1396b3f8c76d  22  99  617.411
# 12  f86b2f28-58b1-4459-93be-1396b3f8c76d  33  88  986.702
# 13  f86b2f28-58b1-4459-93be-1396b3f8c76d  33  99   1951.1
# 14  f86b2f28-58b1-4459-93be-1396b3f8c76d  44  88  1512.02
# 15  f86b2f28-58b1-4459-93be-1396b3f8c76d  44  99  3106.52


import random
from itertools import product
from psweep import psweep as ps


def func(pset):
    return {'result': random.random() * pset['a'] * pset['b']}
            
                
if __name__ == '__main__':
    params1 = ps.loops2params(product(
            ps.seq2dicts('a', [1,2,3,4]),
            ps.seq2dicts('b', [8,9]),
            ))
    params2 = ps.loops2params(product(
            ps.seq2dicts('a', [11,22,33,44]),
            ps.seq2dicts('b', [88,99]),
            ))
    df = ps.run(func, params1)
    # repeat using the same calc_dir reads the database from the previous run
    # and appends to it
    df = ps.run(func, params2)
    print(df)
