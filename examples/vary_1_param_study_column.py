#!/usr/bin/env python3

import random
import psweep as ps


def func(pset):
    return {"result_": random.random() * pset["a"]}


if __name__ == "__main__":
    params = ps.pgrid(ps.plist("a", [1, 2, 3, 4]), [{"study": "a"}])
    df = ps.run(func, params)
    print(df)
