from io import IOBase
import multiprocessing as mp
from functools import partial
import itertools
import os, copy, uuid, pickle, time, shutil
import pandas as pd

pj = os.path.join

# pandas defaults
default_orient = 'records'
pd_time_unit = 's'


#-----------------------------------------------------------------------------
# helpers
#-----------------------------------------------------------------------------

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


# stolen from pwtools and adapted for python3
def is_seq(seq):
    if isinstance(seq, str) or \
       isinstance(seq, IOBase) or \
       isinstance(seq, dict):
        return False
    else:
        try:
            iter(seq)
            return True
        except TypeError:
            return False


def flatten(seq):
    for item in seq:
        if not is_seq(item):
            yield item
        else:
            for subitem in flatten(item):
                yield subitem



#-----------------------------------------------------------------------------
# pandas
#-----------------------------------------------------------------------------


def df_to_json(df, **kwds):
    """Like df.to_json() but with defaults for orient, date_unit, date_format,
    double_precision.

    Parameters
    ----------
    df : pandas.DataFrame
    **kwds :
        passed to df.to_json()
    """
    defaults = dict(
        orient=default_orient,
        date_unit=pd_time_unit,
        date_format='iso',
        double_precision=15)
    for key,val in defaults.items():
        if key not in kwds.keys():
            kwds[key] = val
    return df.to_json(**kwds)


def df_write(df, fn, fmt='pickle', **kwds):
    """Write DataFrame to disk.

    Parameters
    ----------
    df : pandas.DataFrame
    fn : str
        filename
    fmt : str
        {'pickle', 'json'}
    **kwds :
        passed to ``pickle.dump()`` or :func:`df_to_json`
    """
    makedirs(os.path.dirname(fn))
    if fmt == 'pickle':
        with open(fn, 'wb') as fd:
            pickle.dump(df, fd, **kwds)
    elif fmt == 'json':
        df_to_json(df, path_or_buf=fn, **kwds)
    else:
        raise Exception("unknown fmt: {}".format(fmt))


def df_read(fn, fmt='pickle', **kwds):
    """Read DataFrame from file `fn`. See :func:`df_write`."""
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


#-----------------------------------------------------------------------------
# path
#-----------------------------------------------------------------------------

# https://github.com/elcorto/pwtools
def makedirs(path):
    """Create `path` recursively, no questions asked."""
    if not path.strip() == '':
        os.makedirs(path, exist_ok=True)


# https://github.com/elcorto/pwtools
def fullpath(path):
    return os.path.abspath(os.path.expanduser(path))



#-----------------------------------------------------------------------------
# building params
#-----------------------------------------------------------------------------


def plist(name, seq):
    """Create a list of single-item dicts holding the parameter name and a
    value.

    >>> plist('a', [1,2,3])
    [{'a': 1}, {'a': 2}, {'a': 3}]
    """
    return [{name: entry} for entry in seq]

@itr
def merge_dicts(args):
    """Start with an empty dict and update with each arg dict
    left-to-right."""
    dct = {}
    for entry in args:
        dct.update(entry)
    return dct


@itr
def itr2params(loops):
    """Transform the (possibly nested) result of a loop over plists (or
    whatever has been used to create psets) to a proper list of psets
    by flattening and merging dicts.

    Example
    -------
    >>> a = ps.plist('a', [1,2])
    >>> b = ps.plist('b', [77,88])
    >>> c = ps.plist('c', ['const'])

    # result of loops
    >>> list(itertools.product(a,b,c))
    [({'a': 1}, {'b': 77}, {'c': 'const'}),
     ({'a': 1}, {'b': 88}, {'c': 'const'}),
     ({'a': 2}, {'b': 77}, {'c': 'const'}),
     ({'a': 2}, {'b': 88}, {'c': 'const'})]

    # flatten into list of psets
    >>> ps.itr2params(itertools.product(a,b,c))
    [{'a': 1, 'b': 77, 'c': 'const'},
     {'a': 1, 'b': 88, 'c': 'const'},
     {'a': 2, 'b': 77, 'c': 'const'},
     {'a': 2, 'b': 88, 'c': 'const'}]

    # also more nested stuff is no problem
    >>> list(itertools.product(zip(a,b),c))
    [(({'a': 1}, {'b': 77}), {'c': 'const'}),
     (({'a': 2}, {'b': 88}), {'c': 'const'})]

    >>> ps.itr2params(itertools.product(zip(a,b),c))
    [{'a': 1, 'b': 77, 'c': 'const'},
     {'a': 2, 'b': 88, 'c': 'const'}]
    """
    return [merge_dicts(flatten(entry)) for entry in loops]

@itr
def pgrid(plists):
    """Convenience function for the most common loop: nested loops with
    ``itertools.product``: ``ps.itr2params(itertools.product(a,b,c,...))``.

    >>> ps.pgrid(a,b,c)
    [{'a': 1, 'b': 77, 'c': 'const'},
     {'a': 1, 'b': 88, 'c': 'const'},
     {'a': 2, 'b': 77, 'c': 'const'},
     {'a': 2, 'b': 88, 'c': 'const'}]

    >>> ps.pgrid(zip(a,b),c)
    [{'a': 1, 'b': 77, 'c': 'const'},
     {'a': 2, 'b': 88, 'c': 'const'}]
    """
    return itr2params(itertools.product(*plists))


#-----------------------------------------------------------------------------
# run study
#-----------------------------------------------------------------------------

# tmpsave: That's cool, but when running in parallel, we loose the ability to
# store the whole state of the study calculated thus far. For that we would
# need an extra thread that periodically checks for or -- even better -- gets
# informed by workers about finished work and collects the so-far written temp
# results into a global df -- maybe useful for monitoring progress.

def worker_wrapper(pset, worker, tmpsave=False, verbose=False, run_id=None,
                   calc_dir=None, simulate=False):
    """
    Parameters
    ----------
    pset : dict
        example: ``{'a': 1, 'b': 'foo'}``
    run_id : str
        uuid

    See :func:`run` for other parameters.
    """
    assert run_id is not None
    assert calc_dir is not None
    pset_id = str(uuid.uuid4())
    _pset = copy.deepcopy(pset)
    _time_utc = pd.Timestamp(time.time(), unit=pd_time_unit)
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
    if not simulate:
        _pset.update(worker(_pset))
    df_row = pd.DataFrame([_pset], index=[_time_utc])
    if tmpsave:
        fn = pj(calc_dir, 'tmpsave', run_id, pset_id + '.pk')
        df_write(df_row, fn)
    return df_row


def run(worker, params, df=None, poolsize=None, save=True, tmpsave=False,
        verbose=False, calc_dir='calc', backup_script=None,
        backup_calc_dir=False, simulate=False):
    """
    Parameters
    ----------
    worker : callable
        must accept one parameter: `pset` (a dict ``{'a': 1, 'b': 'foo',
        ...}``), return either an update to `pset` or a new dict, result will
        be processes as ``pset.update(worker(pset))``
    params : sequence of dicts
        each dict is a pset ``{'a': 1, 'b': 'foo', ...}``
    df : {pandas.DataFrame, None}
        append rows to this DataFrame, if None then either create new one or
        read existing database file from disk if found
    poolsize : {int, None}
        None : use serial execution
        int : use multiprocessing.Pool (even for ``poolsize=1``)
    save : bool
        save final DataFrame to ``<calc_dir>/results.pk`` (pickle format only)
    tmpsave : bool
        save results from this pset (the current DataFrame row) to
        <calc_dir>/tmpsave/<run_id>/<pset_id>.pk (pickle format only)
    verbose : {bool, sequence}
        | bool : print the current DataFrame row
        | sequence : list of DataFrame column names, print the row but only
        | those columns
    calc_dir : str
    backup_script : {str, None}
        save the file (``backup_script=/path/to/file.py``,
        ``backup_script=__file__``) to ``<calc_dir>/backup_script/<run_id>.py``
    backup_calc_dir : bool
        backup <calc_dir> to <calc_dir>.<timestamp>, where timestamp is derived
        from ``df.index.max()``, i.e. the newest entry on the old database
    simulate : bool
        run everything in <calc_dir>.simulate, don't call `worker`, i.e. save
        what the run would create, but without the results from `worker`,
        useful to check if `params` are correct before starting a production run
    """

    results_fn_base = 'results.pk'

    if simulate:
        calc_dir_sim = calc_dir + '.simulate'
        if os.path.exists(calc_dir_sim):
            shutil.rmtree(calc_dir_sim)
        makedirs(calc_dir_sim)
        old_db = pj(calc_dir, results_fn_base)
        if os.path.exists(old_db):
            shutil.copy(old_db, pj(calc_dir_sim, results_fn_base))
        calc_dir = calc_dir_sim

    results_fn = pj(calc_dir, results_fn_base)

    if df is None:
        if os.path.exists(results_fn):
            df = df_read(results_fn)
        else:
            df = pd.DataFrame()

    if backup_calc_dir and len(df.index) > 0:
        last = df.index.max().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        dst = calc_dir + '_' + last
        assert not os.path.exists(dst), \
            ("backup destination {dst} exists, seems like there has been no new "
             "data in {calc_dir} since the last backup".format(dst=dst,
                                                               calc_dir=calc_dir))
        shutil.copytree(calc_dir, dst)

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
                                     simulate=simulate,
                                     )

    if poolsize is None:
        results = [worker_wrapper_partial(x) for x in params]
    else:
        with mp.Pool(poolsize) as pool:
            results = pool.map(worker_wrapper_partial, params)

    for df_row in results:
        df = df.append(df_row, sort=False)

    if save:
        df_write(df, results_fn)
    return df


#-----------------------------------------------------------------------------
# aliases
#-----------------------------------------------------------------------------
seq2dicts = plist
loops2params = itr2params
