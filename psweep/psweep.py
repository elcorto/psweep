from io import IOBase
from itertools import product
import os, copy
import pandas as pd


def df_json_write(df, name, **kwds):
    orient = kwds.pop('orient', 'records')
    df.to_json(name, double_precision=15, orient=orient, **kwds)


def df_json_read(name, **kwds):
    orient = kwds.pop('orient', 'records')
    return pd.io.json.read_json(name, precise_float=True, orient=orient, **kwds)


def seq2dicts(name, seq):
    """
    >>> seq2dicts('a', [1,2,3])
    [{'a': 1}, {'a': 2}, {'a': 3}]
    """
    return [{name: entry} for entry in seq]


def merge_dicts(lst):
    dct = {}
    for entry in lst:
        dct.update(entry)
    return dct


# stolen from pwtools and adapted for python3
def is_seq(seq):
    if isinstance(seq, str) or \
       isinstance(seq, IOBase) or \
       isinstance(seq, dict):
        return False
    else:
        try:
            _ = iter(seq)
            return True
        except:
            return False


def flatten(seq):
    for item in seq:
        if not is_seq(item):
            yield item
        else:
            for subitem in flatten(item):
                yield subitem


def loops2params(loops):
    return [merge_dicts(flatten(entry)) for entry in loops]


def run(df, func, params, savefn=None, verbose=False):
    runkey = '_run'
    lastrun = df[runkey].values[-1] if runkey in df.columns else -1
    lastidx = -1 if  len(df.index) == 0 else df.index[-1]
    run = lastrun + 1
    for idx,pset in enumerate(params):
        df_row = pd.DataFrame(copy.deepcopy(pset), index=[lastidx + idx + 1])
        if isinstance(verbose, bool) and verbose:
            print(df_row)
        elif is_seq(verbose):
            print(df_row[verbose])
        # update
        df_row[runkey] = run
        for kk,vv in func(copy.deepcopy(pset)).items():
            df_row[kk] = vv
        df = df.append(df_row)
        if savefn:
            _fn = "{savefn}.{run}.{idx}".format(savefn=savefn, run=run, idx=idx)
            df_json_write(df, _fn)
    return df
