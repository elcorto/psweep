#!/usr/bin/env python3

import random
import psweep as ps


def func(pset):
    return {"result_": random.random() * pset["a"] * pset["b"]}


if __name__ == "__main__":
    params = ps.pgrid(
        ps.plist("a", [1, 2, 3]),
        ps.plist("b", [8, 9]),
    )
    # First run.
    df = ps.run(func, params)

    # Second run.
    params = ps.pgrid(
        ps.plist("a", [11, 22, 33]),
        ps.plist("b", [88, 99]),
    )
    df = ps.run(func, params)
    cols = [
        "_run_id",
        "_pset_id",
        "_run_seq",
        "_pset_seq",
        "_pset_hash",
        "a",
        "b",
        "result_",
    ]
    ps.df_print(df[cols])
