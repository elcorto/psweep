#!/usr/bin/env python3

import random
import traceback
from functools import partial

import numpy as np

import psweep as ps


def safe_func(pset, *, func):
    ret = dict()
    try:
        ret.update(func(pset))
        ret.update(_failed=False, _exc_txt=None)
    except:
        txt = traceback.format_exc()
        print(f"failed, traceback:\n{txt}")
        ret.update(_failed=True, _exc_txt=txt)
    finally:
        # some cleanup here if needed
        pass
    return ret


def func_with_fail(pset):
    a = pset["a"]
    # Some fake reason to fail
    if a % 2 == 0:
        raise ValueError("a is even, fail here")
    return {"result_": random.random() * a}


def func_fixed(pset):
    a = pset["a"]
    return {"result_": random.random() * a}


def pset_col_filter(c: str):
    """Filter field names that belong to a pset, so everything *not* starting
    or ending with a "_".

    Example
    -------
    >>> df.columns
    Index(['a', 'b', '_pset_hash', '_pset_id', '_run_seq', '_pset_seq', '_run_id',
           '_calc_dir', '_time_utc', '_exec_host', 'result_', '_pset_runtime'],
           dtype='object')
    >>> list(filter(pset_col_filter, df.columns))
    ['a', 'b']
    """
    return not (c.startswith("_") or c.endswith("_"))


if __name__ == "__main__":
    # Pass a list of ints (i.e. type(1) == int). If we use np.arange(10) then
    # the type of each entry is np.int64 and that leads tp different _pset_hash
    # values, since our hashes are (lukily, sadly?) type-specific due to the
    # usage of joblib.hash().
    params = ps.plist("a", list(range(10)))
    assert isinstance(params[0]["a"], int)
    assert not isinstance(params[0]["a"], np.int64)

    # First run. Don't write df to disk. Pass on here to second run since this
    # is one script. But just using default save=False and letting the second
    # run read it from disk also works of course.

    df = ps.run(
        partial(safe_func, func=func_with_fail),
        params,
        capture_logs="db",
        save=False,
    )
    ps.df_print(df, cols=["a", "result_", "_failed", "_pset_hash", "_run_id"])

    n_failed = len(df[df._failed])
    print(f"{n_failed=}")
    run_id_0 = df._run_id.unique()[0]

    # Repeat failed

    pset_cols = list(filter(pset_col_filter, df.columns))
    print(f"{pset_cols=}")
    params_repeat = [
        row.to_dict() for _, row in df[df._failed][pset_cols].iterrows()
    ]
    print(f"{params_repeat=}")
    df = ps.run(
        partial(safe_func, func=func_fixed),
        params_repeat,
        capture_logs="db",
        df=df,
        save=False,
    )
    ps.df_print(df, cols=["a", "result_", "_failed", "_pset_hash", "_run_id"])
    run_id_1 = df._run_id.unique()[-1]

    assert (
        df[df._run_id == run_id_1].a.values == np.array([0, 2, 4, 6, 8])
    ).all()

    assert (
        df[df._run_id == run_id_1]._pset_hash.values
        == df[df._failed & (df._run_id == run_id_0)]._pset_hash.values
    ).all()
