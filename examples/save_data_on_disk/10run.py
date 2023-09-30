#!/usr/bin/env python3

import os
import subprocess
import psweep as ps


def func(pset):
    fn = os.path.join(pset["_calc_dir"], pset["_pset_id"], "output.txt")
    cmd = (
        f"mkdir -p $(dirname {fn}); "
        f"echo {pset['a']} {pset['a']*2} {pset['a']*4} > {fn}"
    )
    subprocess.run(cmd, shell=True)
    return {"cmd": cmd}


if __name__ == "__main__":
    params = ps.plist("a", [1, 2, 3, 4])
    df = ps.run(func, params)
    print(df)
