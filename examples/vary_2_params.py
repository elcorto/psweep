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
    df = ps.run(func, params)
    print(df)
