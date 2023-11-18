#!/usr/bin/env python3

import random
import sys

import psweep as ps


def func(pset):
    # stdout
    print("text on stdout")

    # stderr
    print("text on stderr", file=sys.stderr)
    return {"result": random.random() * pset["a"]}


if __name__ == "__main__":
    params = ps.plist("a", [1, 2, 3, 4])

    # Write only to disk. No calc/<pset_id>/logs.txt file will be generated.
    ps.run(func, params, capture_logs="db")

    # Only to file
    ps.run(func, params, capture_logs="file")

    # Both
    df = ps.run(func, params, capture_logs="db+file")

    ps.df_print(df, cols=["a", "_pset_id", "_run_id", "_logs"])
