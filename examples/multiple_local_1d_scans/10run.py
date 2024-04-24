#!/usr/bin/env python3

"""
1d scans of either parameter a or b, using constant defaults for the other
from the `const` dict. We use merge_dicts() to combine constant and varied
values. Note that merge_dicts() operates left-to-right, so const must be
left, i.e. replace const values which are varied.
"""

import random
import psweep as ps


def func(pset):
    return {"result_": random.random() * pset["a"] * pset["b"]}


if __name__ == "__main__":
    print("first run")

    const = {"a": 11111, "b": 55555}

    params = []
    disp_cols = []

    values = dict(a=[1, 2, 3], b=[66, 77])

    for study, seq in values.items():
        # [{'a': 1}, {'a': 2}, {'a': 3}]
        # [{'b': 66}, {'b': 77}]
        params_1d = ps.plist(study, seq)

        # [{'a': 1, 'study': 'a'}, {'a': 2, 'study': 'a'}, {'a': 3, 'study': 'a'}]
        # [{'b': 66, 'study': 'b'}, {'b': 77, 'study': 'b'}]
        this_params = ps.pgrid(params_1d, [{"study": study}])

        # [{'a': 1, 'b': 55555, 'study': 'a'}, {'a': 2, 'b': 55555, 'study': 'a'}, {'a': 3, 'b': 55555, 'study': 'a'}]
        # [{'a': 11111, 'b': 66, 'study': 'b'}, {'a': 11111, 'b': 77, 'study': 'b'}]
        this_params = [ps.merge_dicts(const, dct) for dct in this_params]

        params += this_params
        disp_cols.append(study)

    disp_cols += ["_run_id", "study"]
    df = ps.run(func, params, verbose=disp_cols)
    print(df[disp_cols + ["result_"]])
