#!/usr/bin/env python3

import random
import psweep as ps


def func(pset):
    return {'result': random.random() * pset['a'] * pset['b']}


if __name__ == '__main__':
    params = ps.pgrid(
        ps.plist('a', [1,2,3,4]),
        ps.plist('b', [8,9]),
        )
    # First run.
    df = ps.run_local(func, params)

    # Second run.
    params = ps.pgrid(
        ps.plist('a', [11,22,33,44]),
        ps.plist('b', [88,99]),
        )
    df = ps.run_local(func, params)
    print(df)
