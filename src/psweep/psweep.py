from functools import partial
from io import IOBase
import copy
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import pickle
import shutil
import string
import subprocess
import time
import uuid
import warnings
import yaml

import numpy as np
import pandas as pd

pj = os.path.join

# pandas defaults
pandas_default_orient = "records"
pandas_time_unit = "s"


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def system(cmd, **kwds):
    """
    Call shell command.

    Parameters
    ----------
    cmd : str
        shell command
    kwds : dict
        passed to subprocess.run()

    Returns
    -------
    subprocess.CalledProcessError
    """
    try:
        return subprocess.run(
            cmd,
            shell=True,
            check=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            **kwds,
        )
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode())
        raise ex


# https://github.com/elcorto/pwtools
def makedirs(path):
    """Create `path` recursively, no questions asked."""
    if not path.strip() == "":
        os.makedirs(path, exist_ok=True)


# https://github.com/elcorto/pwtools
def fullpath(path):
    return os.path.abspath(os.path.expanduser(path))


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
    if (
        isinstance(seq, str)
        or isinstance(seq, IOBase)
        or isinstance(seq, dict)
    ):
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


def file_write(fn, txt, mode="w"):
    makedirs(os.path.dirname(fn))
    with open(fn, mode=mode) as fd:
        fd.write(txt)


def file_read(fn):
    with open(fn, "r") as fd:
        return fd.read()


def dict_hash(dct, method="sha1"):
    h = getattr(hashlib, method)()
    h.update(json.dumps(dct, sort_keys=True).encode())
    return h.hexdigest()


# -----------------------------------------------------------------------------
# pandas
# -----------------------------------------------------------------------------


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
        orient=pandas_default_orient,
        date_unit=pandas_time_unit,
        date_format="iso",
        double_precision=15,
    )
    for key, val in defaults.items():
        if key not in kwds.keys():
            kwds[key] = val
    return df.to_json(**kwds)


def df_write(df, fn, fmt="pickle", **kwds):
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
    if fmt == "pickle":
        with open(fn, "wb") as fd:
            pickle.dump(df, fd, **kwds)
    elif fmt == "json":
        df_to_json(df, path_or_buf=fn, **kwds)
    else:
        raise Exception("unknown fmt: {}".format(fmt))


def df_read(fn, fmt="pickle", **kwds):
    """Read DataFrame from file `fn`. See :func:`df_write`."""
    if fmt == "pickle":
        with open(fn, "rb") as fd:
            return pickle.load(fd, **kwds)
    elif fmt == "json":
        orient = kwds.pop("orient", pandas_default_orient)
        return pd.io.json.read_json(
            fn, precise_float=True, orient=orient, **kwds
        )
    else:
        raise Exception("unknown fmt: {}".format(fmt))


def df_print(df, index=False):
    """Print DataFrame, by default without the index."""
    print(df.to_string(index=index))


def df_filter_conds(df, conds):
    """Filter DataFrame using bool arrays/Series/DataFrames in `conds`.

    Logical-and all bool sequences in `conds`. Same as

    >>> df[conds[0] & conds[1] & conds[2] & ...]

    but `conds` can be programatically generated while the expression above
    would need to be changed by hand if `conds` changes.

    Parameters
    ----------
    df : DataFrame
    conds : sequence
        list of bool masks

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> df=pd.DataFrame({'a': arange(10), 'b': arange(10)+4})
    >>> c1=df.a > 3
    >>> c2=df.b < 9
    >>> c3=df.a % 2 == 0
    >>> df[c1 & c2 & c3]
       a  b
    4  4  8
    >>> ps.df_filter_conds(df, [c1, c2, c3])
       a  b
    4  4  8
    """
    if len(conds) == 0:
        msk = [True] * len(df)
    elif len(conds) == 1:
        msk = conds[0]
    else:
        msk = np.logical_and.reduce(conds)
    return df[msk]


# -----------------------------------------------------------------------------
# building params
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# run study
# -----------------------------------------------------------------------------

# tmpsave: That's cool, but when running in parallel, we loose the ability to
# store the whole state of the study calculated thus far. For that we would
# need an extra thread that periodically checks for or -- even better -- gets
# informed by workers about finished work and collects the so-far written temp
# results into a global df -- maybe useful for monitoring progress.


def worker_wrapper(
    pset,
    worker,
    tmpsave=False,
    verbose=False,
    run_id=None,
    calc_dir=None,
    simulate=False,
    pset_seq=np.nan,
    run_seq=None,
):
    """
    Parameters
    ----------
    pset : dict
        example: ``{'a': 1, 'b': 'foo'}``
    run_id : str
        uuid

    See :func:`run_local` for other parameters.
    """
    assert run_id is not None
    assert calc_dir is not None
    pset_id = str(uuid.uuid4())
    _pset = copy.deepcopy(pset)
    _time_utc = pd.Timestamp(time.time(), unit=pandas_time_unit)
    hash_alg = "sha1"
    try:
        pset_hash = dict_hash(pset, hash_alg)
    except TypeError:
        pset_hash = np.nan
    update = {
        "_run_id": run_id,
        "_pset_id": pset_id,
        "_calc_dir": calc_dir,
        "_time_utc": _time_utc,
        f"_pset_{hash_alg}": pset_hash,
        "_pset_seq": pset_seq,
        "_run_seq": run_seq,
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
        fn = pj(calc_dir, "tmpsave", run_id, pset_id + ".pk")
        df_write(df_row, fn)
    return df_row


def run_local(
    worker,
    params,
    df=None,
    poolsize=None,
    save=True,
    tmpsave=False,
    verbose=False,
    calc_dir="calc",
    simulate=False,
    database_dir=None,
    database_basename="database.pk",
    backup=False,
    backup_calc_dir=False,
):
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
        save final DataFrame to ``<calc_dir>/database.pk`` (pickle format only)
    tmpsave : bool
        save results from this pset (the current DataFrame row) to
        <calc_dir>/tmpsave/<run_id>/<pset_id>.pk (pickle format only)
    verbose : {bool, sequence}
        | bool : print the current DataFrame row
        | sequence : list of DataFrame column names, print the row but only
        | those columns
    calc_dir : str
    simulate : bool
        run everything in <calc_dir>.simulate, don't call `worker`, i.e. save
        what the run would create, but without the results from `worker`,
        useful to check if `params` are correct before starting a production run
    database_dir : str
        Path for the database. Default is <calc_dir>.
    database_basename : str
        <database_dir>/<database_basename>
    backup : bool
        Make backup of <calc_dir> to <calc_dir>.bak_<timestamp>_run_id_<_run_id_>
    """

    database_dir = calc_dir if database_dir is None else database_dir

    if simulate:
        calc_dir_sim = calc_dir + ".simulate"
        if os.path.exists(calc_dir_sim):
            shutil.rmtree(calc_dir_sim)
        makedirs(calc_dir_sim)
        old_db = pj(database_dir, database_basename)
        if os.path.exists(old_db):
            shutil.copy(old_db, pj(calc_dir_sim, database_basename))
        else:
            warnings.warn(f"simulate: {old_db} not found, will create new db in "
                          f"{calc_dir_sim}")
        database_fn = pj(calc_dir_sim, database_basename)
        calc_dir = calc_dir_sim
    else:
        database_fn = pj(database_dir, database_basename)

    if df is None:
        if os.path.exists(database_fn):
            df = df_read(database_fn)
        else:
            df = pd.DataFrame()

    if len(df) == 0:
        pset_seq_old = -1
        run_seq_old = -1
    else:
        pset_seq_old = df._pset_seq.values.max()
        run_seq_old = df._run_seq.values.max()


    if backup_calc_dir:
        warnings.warn(
            "'backup_calc_dir' was renamed to 'backup'", DeprecationWarning
        )
        backup = True
    if backup and len(df.index) > 0:
        stamp = df.index.max().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        dst = f"{calc_dir}.bak_{stamp}_run_id_{df._run_id.values[-1]}"
        assert not os.path.exists(dst), (
            "backup destination {dst} exists, seems like there has been no new "
            "data in {calc_dir} since the last backup".format(
                dst=dst, calc_dir=calc_dir
            )
        )
        shutil.copytree(calc_dir, dst)

    run_id = str(uuid.uuid4())

    worker_wrapper_partial = partial(
        worker_wrapper,
        worker=worker,
        tmpsave=tmpsave,
        verbose=verbose,
        run_id=run_id,
        calc_dir=calc_dir,
        simulate=simulate,
    )

    if poolsize is None:
        results = [
            worker_wrapper_partial(
                pset=pset,
                pset_seq=pset_seq_old + ii + 1,
                run_seq=run_seq_old + 1,
            )
            for ii, pset in enumerate(params)
        ]
    else:
        # Can't use lambda here b/c pool.map() still can't pickle local scope
        # lambdas. That's why we emulate
        #   pool.map(lambda pset: worker_wrapper_partial(pset, run_seq=...,
        #            params)
        # with nested partial(). Cool, eh?
        _worker_wrapper_partial = partial(
            worker_wrapper_partial, run_seq=run_seq_old + 1
        )
        with mp.Pool(poolsize) as pool:
            results = pool.map(_worker_wrapper_partial, params)

    for df_row in results:
        df = df.append(df_row, sort=False)

    if save:
        df_write(df, database_fn)

    return df


# -----------------------------------------------------------------------------
# HPC cluster batch runs
# -----------------------------------------------------------------------------


class Machine:
    def __init__(self, machine_dir, jobscript_name="jobscript"):
        # templates/machines/<name>/info.yaml
        # ^^^^^^^^^^^^^^^^^^^^^^^^^------------- machine_dir
        # templates/machines/<name>/jobscript
        #                           ^^^^^^^^^--- template.basename
        self.name = os.path.basename(os.path.normpath(machine_dir))
        self.template = FileTemplate(
            pj(machine_dir, jobscript_name), target_suffix="_" + self.name
        )
        with open(pj(machine_dir, "info.yaml")) as fd:
            info = yaml.safe_load(fd)
        for key, val in info.items():
            assert key not in self.__dict__, f"cannot overwrite '{key}'"
            setattr(self, key, val)

    def __repr__(self):
        return f"{self.name}:{self.template}"


class FileTemplate:
    def __init__(self, filename, target_suffix=""):
        self.filename = filename
        self.basename = os.path.basename(filename)
        self.dirname = os.path.dirname(filename)
        self.targetname = f"{self.basename}{target_suffix}"

    def __repr__(self):
        return self.filename

    def fill(self, pset):
        txt = file_read(self.filename)
        return string.Template(txt).substitute(pset)


def gather_calc_templates(calc_templ_dir):
    return [
        FileTemplate(pj(calc_templ_dir, basename))
        for basename in os.listdir(calc_templ_dir)
    ]


def gather_machines(machine_templ_dir):
    return [
        Machine(pj(machine_templ_dir, basename))
        for basename in os.listdir(machine_templ_dir)
    ]


def git_clean():
    cmd = "git status --porcelain"
    return system(cmd).stdout.decode() == ""


# If we ever add a "simulate" kwd here: don't pass that thru to run_local() b/c
# there this prevents worker() from being executed, but that's what we always
# want here since it writes only input files. Instead, just set calc_dir =
# calc_dir_sim and copy the database as in run_local() and go. Don't copy the
# run_*.sh scripts b/c they are generated afresh anyway.
#
def prep_batch(
    params,
    calc_dir="calc",
    calc_templ_dir="templates/calc",
    machine_templ_dir="templates/machines",
    git=False,
    backup=False,
):

    if git:
        if not os.path.exists(".git"):
            system("git init; git add -A; git commit -m 'psweep: init'")
        if not git_clean():
            raise Exception("dirty git repo")

    calc_templates = gather_calc_templates(calc_templ_dir)
    machines = gather_machines(machine_templ_dir)
    templates = calc_templates + [m.template for m in machines]

    def worker(pset):
        for template in templates:
            file_write(
                pj(calc_dir, pset["_pset_id"], template.targetname),
                template.fill(pset),
            )
        return {}

    df = run_local(worker, params, calc_dir=calc_dir, backup=backup)

    msk_latest = df._run_seq == df._run_seq.values.max()
    msk_old = df._run_seq < df._run_seq.values.max()
    for machine in machines:
        txt = ""
        for pfx, msk in [("# ", msk_old), ("", msk_latest)]:
            if msk.any():
                txt += "\n"
            txt += "\n".join(
                f"{pfx}cd {pset_id}; {machine.subcmd} {machine.template.targetname}; cd $here  # run_seq={run_seq} pset_seq={pset_seq}"
                for pset_id, pset_seq, run_seq in zip(
                    df[msk]._pset_id.values,
                    df[msk]._pset_seq.values,
                    df[msk]._run_seq.values,
                )
            )
        file_write(
            f"{calc_dir}/run_{machine.name}.sh",
            f"#!/bin/sh\n\nhere=$(pwd)\n{txt}\n",
        )

    if git:
        if not git_clean():
            system(
                f"git add -A; git commit -m 'psweep: run_id={df._run_id.values[-1]}'"
            )
    return df


# -----------------------------------------------------------------------------
# aliases
# -----------------------------------------------------------------------------
seq2dicts = plist
loops2params = itr2params


def run(*args, **kwds):
    warnings.warn("run() was renamed to run_local()", DeprecationWarning)
    return run_local(*args, **kwds)
