#!/usr/bin/env python3

# ForkPoolWorker-1
# ForkPoolWorker-2
# ForkPoolWorker-1
# ForkPoolWorker-2
# ForkPoolWorker-1
# ForkPoolWorker-2
# ForkPoolWorker-2
# ForkPoolWorker-1
#   _calc_dir                              _pset_id  \
# 0      calc  6a295117-4af6-47ec-84fd-16e2693ec7fe   
# 1      calc  e0cd852b-5212-4c11-89a5-2e6014fea03d   
# 2      calc  979399be-ae86-4918-9d2a-1d35eea1bb6a   
# 3      calc  4c26d33e-4fdc-479c-9685-4100b423b17f   
# 4      calc  0a89ff32-e5cc-43a1-ae8d-813cd4ea19ae   
# 5      calc  684baa64-60f0-4ffe-87b3-1bdf23718522   
# 6      calc  44b8089e-c4bc-4c3c-885e-cdcdd6296d21   
# 7      calc  63a4459f-beda-41c6-8ef6-7cd33ac5d180   
# 
#                                 _run_id  a    result  
# 0  baaf7291-73cb-48ce-819c-151451be2135  1  0.842526  
# 1  baaf7291-73cb-48ce-819c-151451be2135  2   1.49021  
# 2  baaf7291-73cb-48ce-819c-151451be2135  3  0.269487  
# 3  baaf7291-73cb-48ce-819c-151451be2135  4  0.233928  
# 4  baaf7291-73cb-48ce-819c-151451be2135  5   1.58186  
# 5  baaf7291-73cb-48ce-819c-151451be2135  6   3.73453  
# 6  baaf7291-73cb-48ce-819c-151451be2135  7   6.52404  
# 7  baaf7291-73cb-48ce-819c-151451be2135  8   2.32329  


import random
import multiprocessing as mp
from psweep import psweep as ps

def func(pset):
    print(mp.current_process().name)
    return {'result': random.random() * pset['a']}
                
if __name__ == '__main__':
    params = ps.seq2dicts('a', [1,2,3,4,5,6,7,8])
    df = ps.run(func, params, poolsize=2)
    print(df)
