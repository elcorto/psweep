#!/usr/bin/env python3

import random
import psweep as ps


def func(pset):
    return {'result': random.random() * pset['a'] * pset['b']}


if __name__ == '__main__':
    sel = ['_calc_dir', 'a', 'b', 'result']

    # 1st real run: produce some data, vary a, b constant
    params = ps.pgrid(
        ps.plist('a', [1,2,3,4]),
        ps.plist('b', [100]))
    df = ps.run(func, params)
    print(df[sel])

    # simulate run: check if new parameter grid is OK
    params = ps.pgrid(
        ps.plist('a', [5,6]),
        ps.plist('b', [88, 99]))
    df = ps.run(func, params, simulate=True)
    print(df[sel])

    # looks good, 2nd real run with new params
    df = ps.run(func, params)
    print(df[sel])
