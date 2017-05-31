#!/usr/bin/env python3

"""Example for benchmarking a shell command, depending on parameters a
and b.

params:

[{'a': 1, 'const': 'const', 'b': 1.0}, 
 {'a': 1, 'const': 'const', 'b': 2.0}, 
 {'a': 2, 'const': 'const', 'b': 1.0}, 
 {'a': 2, 'const': 'const', 'b': 2.0}, 
 {'a': 3, 'const': 'const', 'b': 1.0}, 
 {'a': 3, 'const': 'const', 'b': 2.0}, 
 {'a': 4, 'const': 'const', 'b': 1.0}, 
 {'a': 4, 'const': 'const', 'b': 2.0}]

results:

$ jq . results.json                                                                                                                                          
{
  "columns": [
    "_run",
    "a",
    "b",
    "const",
    "timing"
  ],
  "index": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
  ],
  "data": [
    [
      0,
      1,
      1,
      "const",
      0.103926325
    ],
    [
      0,
      1,
      2,
      "const",
      0.055555148
    ],
    ....
"""

import timeit, os
from itertools import product
from subprocess import run
from psweep import psweep as ps
import pandas as pd
        

def func(pset):
    cmd_tmpl = """
        echo "calling func: pset: a={a} b={b} const={const}"; 
        sleep $(echo "0.1 * {a} / {b}" | bc -l)
        """
    cmd = cmd_tmpl.format(**pset)
    timing = min(timeit.repeat(lambda: run(cmd, shell=True),
                               repeat=3,
                               number=1))
    return {'timing': timing}
            
                
if __name__ == '__main__':
    results = 'results.json'
    params = ps.loops2params(product(
            ps.seq2dicts('a', [1,2,3,4]),
            ps.seq2dicts('b', [1.0,2.0]),
            ps.seq2dicts('const', ['const']),
            ))
    # Continue to write old json database if present. Use orient='split' b/c of
    # https://github.com/pandas-dev/pandas/issues/12866 (default orient setting
    # converts DataFrame.index int -> str in to_json(), reading back in
    # converts str -> float).
    if os.path.exists(results):
        df = pd.io.json.read_json(results, orient='split')
    else:
        df = pd.DataFrame()
    df = ps.run(df, func, params, savefn='save.json')
    df.to_json(results, orient='split')
