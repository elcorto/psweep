#!/usr/bin/env python3

import os
import subprocess
import psweep as ps


def func(pset):
    fn = os.path.join(pset['_calc_dir'],
                      pset['_pset_id'],
                      'output.txt')
    cmd = "mkdir -p $(dirname {fn}); echo {a} > {fn}".format(a=pset['a'],
                                                             fn=fn)
    pset['cmd'] = cmd
    subprocess.run(cmd, shell=True)
    return pset


if __name__ == '__main__':
    params = ps.plist('a', [1,2,3,4])
    df = ps.run(func, params)
    print(df)
