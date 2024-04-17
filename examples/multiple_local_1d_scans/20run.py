#!/usr/bin/env python3

import random
import psweep as ps


def func(pset):
    return {"result_": random.random() * pset["a"] * pset["b"]}


if __name__ == "__main__":
    print("second run")

    const = {"a": 11111, "b": 55555}

    params = []
    disp_cols = []

    values = dict(b=[88, 99])

    for study, seq in values.items():
        params_1d = ps.plist(study, seq)
        this_params = ps.pgrid(params_1d, [{"study": study}])
        this_params = [ps.merge_dicts(const, dct) for dct in this_params]
        params += this_params
        disp_cols.append(study)

    disp_cols += ["_run_id", "study"]
    df = ps.run(func, params, verbose=disp_cols, backup=True)
    print(df[disp_cols + ["result_"]])
