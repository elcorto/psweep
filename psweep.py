from io import IOBase
from itertools import product
import os, copy
import pandas as pd


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


def run(df, func, params, savefn=None):
    runkey = '_run'
    lastrun = df[runkey].values[-1] if runkey in df.columns else -1
    lastidx = -1 if  len(df.index) == 0 else df.index[-1]
    run = lastrun + 1
    for idx,dct in enumerate(params):
        row = copy.deepcopy(dct)
        row.update(func(dct))
        row.update({'_run': run})
        df_row = pd.DataFrame(row, index=[lastidx + idx + 1])
        print(df_row)
        df = df.append(df_row)
        if savefn:
            _fn = "{savefn}.{run}.{idx}".format(savefn=savefn, run=run, idx=idx)
            df.to_json(_fn, orient='split')
    return df
