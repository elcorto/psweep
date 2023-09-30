#!/usr/bin/env python3

import random
import psweep as ps


def func(pset):
    return {"result": random.random() * pset["a"]}


if __name__ == "__main__":
    params = ps.plist("a", [1, 2, 3])

    # First run.
    ps.run(func, params)

    # Second run.
    df = ps.run(func, params)

    cols = [
        "_run_id",
        "_pset_id",
        "_run_seq",
        "_pset_seq",
        "_pset_hash",
        "a",
        "result",
    ]
    ps.df_print(df[cols])
