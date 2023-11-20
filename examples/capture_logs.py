#!/usr/bin/env python3

import random
import sys
import subprocess

import psweep as ps


def call(cmd: str):
    """Execute shell command and return any text output on stdout or stderr."""
    # check=False: don't raise exception, only capture shell error text
    return subprocess.run(
        cmd,
        shell=True,
        check=False,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
    ).stdout.decode()


def func(pset):
    # stdout
    print("text on stdout")
    print(call("ls -l | head -n4"))

    # This will be output on stderr by the shell interpreter, but since (i) we
    # merge stdout and stderr in call() and (ii) use print() which outputs on
    # stdout, this will end up there.
    print(call("xxxyyyzzz will raise shell error"))

    # stderr
    print("text on stderr", file=sys.stderr)
    return {"result": random.random() * pset["a"]}


if __name__ == "__main__":
    params = ps.plist("a", [1, 2, 3, 4])

    # Write only to db. No calc/<pset_id>/logs.txt file will be generated.
    df = ps.run(func, params, capture_logs="db")
    print(df._logs.values[-1])

    # Only to file, print last disk file content from this run
    df = ps.run(func, params, capture_logs="file")
    pset_id = df._pset_id.values[-1]
    # calc_dir is the same for all psets, so just grab the first one
    calc_dir = df._calc_dir.values[0]
    # These do the same
    ##print(call(f"cat {calc_dir}/{pset_id}/logs.txt"))
    print(ps.file_read(f"{calc_dir}/{pset_id}/logs.txt"))

    # Both
    df = ps.run(func, params, capture_logs="db+file")
