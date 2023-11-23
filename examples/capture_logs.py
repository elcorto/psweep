#!/usr/bin/env python3

import random
import sys
import subprocess
import traceback
import textwrap

import psweep as ps


def call(cmd: str, raise_error=False):
    """Execute shell command and return any text output on stdout or stderr.

    If raise_error=True then let subprocess raise a
    subprocess.CalledProcessError exception if the shell exits != 0.
    """
    # check=False: don't raise exception, only capture shell error text
    return subprocess.run(
        cmd,
        shell=True,
        check=raise_error,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
    ).stdout.decode()


def func_no_exc(pset):
    # stdout
    print("text on stdout")
    print(call("ls -l | head -n4"))

    # This will be output on stderr by the shell interpreter, but since (i) we
    # merge stdout and stderr in call() and (ii) use print() which outputs on
    # stdout, this will end up there.
    print(call("xxyyzz this will raise a shell error", raise_error=False))

    # stderr
    print("text on stderr", file=sys.stderr)
    return {"result": random.random() * pset["a"]}


def func_with_exc(pset):
    # stdout
    print("text on stdout")
    print(call("ls -l | head -n4"))

    # This will raise CalledProcessError, so we need to wrap this in
    # safe_func().
    print(
        call(
            "xxyyzz this will raise a shell error and an exception",
            raise_error=True,
        )
    )
    return {"result": random.random() * pset["a"]}


def safe_func(pset):
    try:
        return func_with_exc(pset)
    except:
        print(traceback.format_exc())
        return dict()


def msg(header, txt):
    h = textwrap.dedent(
        f"""
        {"-"*79}
        {header}
        {"-"*79}
        """
    )
    print(f"{h}\n{txt}")


if __name__ == "__main__":
    params = ps.plist("a", [1, 2, 3, 4])

    for func in [func_no_exc, safe_func]:
        # Write only to db. No calc/<pset_id>/logs.txt file will be generated.
        df = ps.run(func, params, capture_logs="db")
        msg("log entry from db", df._logs.values[-1])

        # Only to file, print last disk file content from this run
        df = ps.run(func, params, capture_logs="file")
        pset_id = df._pset_id.values[-1]
        # calc_dir is the same for all psets, so just grab the first one
        calc_dir = df._calc_dir.values[0]
        # These do the same
        ##print(call(f"cat {calc_dir}/{pset_id}/logs.txt"))
        msg(
            "content of one log file",
            ps.file_read(f"{calc_dir}/{pset_id}/logs.txt"),
        )

        # Both
        df = ps.run(func, params, capture_logs="db+file")
