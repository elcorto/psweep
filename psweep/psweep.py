from io import IOBase
import multiprocessing as mp
from functools import partial
import os, copy, uuid, pickle, time, shutil
import pandas as pd

pj = os.path.join


default_orient = 'records'


def df_write(df, fn, fmt='pickle', **kwds):
    makedirs(os.path.dirname(fn))
    if fmt == 'pickle':
        with open(fn, 'wb') as fd:
            pickle.dump(df, fd, **kwds)
    elif fmt == 'json':
        orient = kwds.pop('orient', default_orient)
        df.to_json(fn, double_precision=15, orient=orient, **kwds)
    else:
        raise Exception("unknown fmt: {}".format(fmt))


def df_read(fn, fmt='pickle', **kwds):
    if fmt == 'pickle':
        with open(fn, 'rb') as fd:
            return pickle.load(fd, **kwds)
    elif fmt == 'json':
        orient = kwds.pop('orient', default_orient)
        return pd.io.json.read_json(fn,
                                    precise_float=True,
                                    orient=orient,
                                    **kwds)
    else:
        raise Exception("unknown fmt: {}".format(fmt))


# https://github.com/elcorto/pwtools
def makedirs(path):
    if not path.strip() == '':
        os.makedirs(path, exist_ok=True)


# https://github.com/elcorto/pwtools
def fullpath(path):
    return os.path.abspath(os.path.expanduser(path))


def seq2dicts(name, seq):
    """
    >>> seq2dicts('a', [1,2,3])
    [{'a': 1}, {'a': 2}, {'a': 3}]
    """
    return [{name: entry} for entry in seq]


def itr(func):
    """Decorator which makes functions take a sequence of args or individual
    args.

    ::

        @itr
        def func(seq):
            for arg in seq:
                ...
        @itr
        def func(*args):
            for arg in args:
                ...
    """

    def wrapper(*args):
        if len(args) == 1:
            return func(args[0])
        else:
            return func(args)
    return wrapper


@itr
def merge_dicts(args):
    """Start with an empty dict and update with each arg dict
    left-to-right."""
    dct = {}
    for entry in args:
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


@itr
def loops2params(loops):
    return [merge_dicts(flatten(entry)) for entry in loops]


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
    _pset = copy.deepcopy(pset)
    _time_utc = pd.Timestamp(time.time(), unit='s')
    update = {'_run_id': run_id,
              '_pset_id': pset_id,
              '_calc_dir': calc_dir,
              '_time_utc': _time_utc
              }
    _pset.update(update)
    # for printing only
    df_row = pd.DataFrame([_pset])
    if isinstance(verbose, bool) and verbose:
        print(df_row)
    elif is_seq(verbose):
        print(df_row[verbose])
    _pset.update(worker(_pset))
    df_row = pd.DataFrame([_pset], index=[_time_utc])
    if tmpsave:
        fn = pj(calc_dir, 'tmpsave', run_id, pset_id + '.pk')
        df_write(df_row, fn)
    return df_row


def run(worker, params, df=None, poolsize=None, save=True, tmpsave=False,
        verbose=False, calc_dir='calc', backup_script=None):

    results_fn = pj(calc_dir, 'results.pk')

    if df is None:
        if os.path.exists(results_fn):
            df = df_read(results_fn)
        else:
            df = pd.DataFrame()

    run_id = str(uuid.uuid4())
    if backup_script is not None:
        assert os.path.exists(backup_script), \
            "{} does not exist".format(backup_script)

        path = pj(calc_dir, 'backup_script')
        makedirs(path)
        shutil.copy(backup_script , pj(path, run_id + '.py'))

    worker_wrapper_partial = partial(worker_wrapper,
                                     worker=worker,
                                     tmpsave=tmpsave,
                                     verbose=verbose,
                                     run_id=run_id,
                                     calc_dir=calc_dir,
                                     )

    if poolsize is None:
        results = [worker_wrapper_partial(x) for x in params]
    else:
        with mp.Pool(poolsize) as pool:
            results = pool.map(worker_wrapper_partial, params)

    for df_row in results:
        df = df.append(df_row)

    if save:
        df_write(df, results_fn)
    return df
