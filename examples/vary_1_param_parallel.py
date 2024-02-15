#!/usr/bin/env python3

import random
import multiprocessing as mp
import psweep as ps


def func(pset):
    print(mp.current_process().name)
    return {"result": random.random() * pset["a"]}


if __name__ == "__main__":
    params = ps.plist("a", [1, 2, 3, 4, 5, 6, 7, 8])
    df = ps.run(func, params, poolsize=2)
    print(df)
