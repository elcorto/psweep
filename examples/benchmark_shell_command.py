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
"""

import timeit
from subprocess import run
import psweep as ps


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
    params = ps.pgrid(
        ps.plist('a', [1,2,3,4]),
        ps.plist('b', [1.0,2.0]),
        ps.plist('const', ['const']),
        )
    df = ps.run(func, params)
