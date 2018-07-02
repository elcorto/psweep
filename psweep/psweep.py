from io import IOBase
from itertools import product
import multiprocessing as mp
from functools import partial
import os, copy, uuid
import pandas as pd

pj = os.path.join


dtype_err_msg = ("reading and writing the json with dtype other than object "
                 "will allow Pandas typecasting, which we very much dislike")


def df_json_write(df, name, **kwds):
    assert (df.dtypes == object).all(), dtype_err_msg
    orient = kwds.pop('orient', 'records')
    df.to_json(name, double_precision=15, orient=orient, **kwds)


def df_json_read(name, **kwds):
    orient = kwds.pop('orient', 'records')
    dtype = kwds.pop('dtype', object)
    if dtype is not object:
        raise ValueError(dtype_err_msg)
    return pd.io.json.read_json(name, 
                                precise_float=True, 
                                orient=orient, 
                                dtype=dtype,
                                **kwds)


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


def run_serial(df, func, params, tmpsave=None, verbose=False):
    runkey = '_run'
    lastrun = df[runkey].values[-1] if runkey in df.columns else -1
    lastidx = -1 if  len(df.index) == 0 else df.index[-1]
    run = lastrun + 1
    for idx,pset in enumerate(params):
        index = lastidx + idx + 1
        rowopts = dict(index=[index], dtype=object)
        df_row = pd.DataFrame(copy.deepcopy(pset), **rowopts)
        if isinstance(verbose, bool) and verbose:
            print(df_row)
        elif is_seq(verbose):
            print(df_row[verbose])
        row_update = {runkey: run}
        row_update.update(func(copy.deepcopy(pset)))
        df_row = pd.concat([df_row, 
                            pd.DataFrame([row_update], **rowopts)], axis=1)
        df = df.append(df_row)
        if tmpsave:
            _fn = "{tmpsave}.{run}.{idx}".format(tmpsave=tmpsave, run=run, idx=idx)
            df_json_write(df, _fn)
    return df

# tmpsave: That's cool, but when running in parallel, we loose the ability to
# store the whole state of the study calculated thus far. For that we would
# need an extra thread that periodically checks for or -- even better -- gets
# informed by workers about finished work and collects the so-far written temp
# results into a global df -- maybe useful for monitoring progress.

def worker_wrapper(pset, worker, tmpsave=False, verbose=False, run_id=None,
                   calc_dir=None):
    assert run_id is not None
    assert calc_dir is not None
    pset_id = str(uuid.uuid4())
    # for printing only
    df_row = pd.DataFrame([pset], dtype=object)
    if isinstance(verbose, bool) and verbose:
        print(df_row)
    elif is_seq(verbose):
        print(df_row[verbose])
    _pset = copy.deepcopy(pset)
    update = {'_run_id': run_id,
              '_pset_id': pset_id,
              '_calc_dir': calc_dir,
              }
    _pset.update(update)
    _pset.update(worker(_pset))
    df_row = pd.DataFrame([_pset], dtype=object)
    if tmpsave:
        fn = pj(calc_dir, 'tmpsave', run_id, pset_id + '.json')
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        df_json_write(df_row, fn)
    return df_row
    

def run(worker, params, df=None, poolsize=1, tmpsave=False, verbose=False,
        calc_dir='calc'):
    results_fn = pj(calc_dir, 'results.json')
    os.makedirs(calc_dir, exist_ok=True)
    
    if df is None:
        if os.path.exists(results_fn):
            # TODO: backup .. etc, all the usual boilerplate
            df = df_json_read(results_fn)
        else:
            df = pd.DataFrame()

    run_id = str(uuid.uuid4())
    worker_wrapper_partial = partial(worker_wrapper,
                                     worker=worker,
                                     tmpsave=tmpsave,
                                     verbose=verbose,
                                     run_id=run_id,
                                     calc_dir=calc_dir,
                                     )
    with mp.Pool(poolsize) as pool:
        results = pool.map(worker_wrapper_partial, params)
    
    for df_row in results:
        df = df.append(df_row, ignore_index=True)

    df_json_write(df, results_fn)
    return df
