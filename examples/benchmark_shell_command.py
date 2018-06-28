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
[
  {
    "a": 1,
    "b": 1,
    "_run": 0,
    "timing": 0.894392793854928
  },
  {
    "a": 1,
    "b": 2,
    "_run": 0,
    "timing": 1.615922232263486
  },
  {
    "a": 2,
    "b": 1,
    "_run": 0,
    "timing": 1.026592040160289
  },
  {
    "a": 2,
    "b": 2,
    "_run": 0,
    "timing": 0.355394254328504
  },
  {
    "a": 3,
    "b": 1,
    "_run": 0,
    "timing": 1.865790378600461
  },
  {
    "a": 3,
    "b": 2,
    "_run": 0,
    "timing": 1.029431201749784
  },
  {
    "a": 4,
    "b": 1,
    "_run": 0,
    "timing": 1.861546694473383
  },
  {
    "a": 4,
    "b": 2,
    "_run": 0,
    "timing": 0.740852334581327
  }
]
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
        df = ps.df_json_read(results)
    else:
        df = pd.DataFrame()
    df = ps.run(df, func, params, tmpsave='save.json')
    ps.df_json_write(df, results)
