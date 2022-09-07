from functools import partial, wraps
from io import IOBase
from typing import Any, Union, Sequence, Callable, Iterator
import copy
import itertools
import multiprocessing as mp
import os
import pickle
import re
import shutil
import string
import subprocess
import time
import uuid
import warnings
import yaml
import sys

import numpy as np
import pandas as pd

import joblib

pj = os.path.join

# defaults, globals
PANDAS_DEFAULT_ORIENT = "records"
PANDAS_TIME_UNIT = "s"
PSET_HASH_ALG = "sha1"
GIT_ADD_ALL = "git add -A -v"


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def system(cmd: str, **kwds) -> subprocess.CompletedProcess:
    """
    Call shell command.

    Parameters
    ----------
    cmd :
        shell command
    kwds :
        keywords passed to `subprocess.run`
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
def makedirs(path: str) -> None:
    """Create `path` recursively, no questions asked."""
    if not path.strip() == "":
        os.makedirs(path, exist_ok=True)


# https://github.com/elcorto/pwtools
def fullpath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def itr(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args):
        # (arg1,)
        if len(args) == 1:
            arg = args[0]
            return func(arg if is_seq(arg) else [arg])
        # (arg1,...,argN)
        else:
            return func(args)

    return wrapper


# https://github.com/elcorto/pwtools
def is_seq(seq) -> bool:
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


def file_write(fn: str, txt: str, mode="w"):
    makedirs(os.path.dirname(fn))
    with open(fn, mode=mode) as fd:
        fd.write(txt)


def file_read(fn: str):
    with open(fn, "r") as fd:
        return fd.read()


class PsweepHashError(TypeError):
    pass


def pset_hash(
    dct: dict,
    method=PSET_HASH_ALG,
    raise_error=True,
    skip_special_cols=True,
):
    """Reproducible hash of a dict for usage in database (hash of a `pset`)."""

    # We target "reproducible" hashes, i.e. not what Python's ``hash`` function
    # does, for instance for two interpreter sessions::
    #
    #     $ python
    #     >>> hash("12")
    #     8013944793133897043
    #
    #     $ python
    #     >>> hash("12")
    #     4021864388667373027
    #
    # We can't hash, say, the pickled byte string of some object (e.g.
    # ``hash(pickle.dumps(obj))``), b/c that may contain a ref to its memory
    # location which is not what we're interested in. Similarly, also using
    # ``repr`` is not reproducible::
    #
    #     >>> class Foo:
    #     ...     pass
    #
    #     >>> repr(Foo())
    #     '<__main__.Foo object at 0x7fcc68aa9d60>'
    #     >>> repr(Foo())
    #     '<__main__.Foo object at 0x7fcc732034c0>'
    #
    # even though for our purpose, we'd consider the two instances of ``Foo`` to
    # be the same.
    #
    # The same observations have been also made elsewhere [1,2]. Esp. [2]
    # points to [3] which in turn mentions joblib.hashing.hash(). It's code
    # shows how complex the problem is, but so far this is our best bet.
    #
    # [1] https://death.andgravity.com/stable-hashing
    # [2] https://ourpython.com/python/deterministic-recursive-hashing-in-python
    # [3] https://stackoverflow.com/a/52175075
    if skip_special_cols:
        _dct = {
            key: val for key, val in dct.items() if not key.startswith("_")
        }
    else:
        _dct = dct
    # joblib can hash "anything" so we didn't come up with an input that
    # actually fails to hash. As such, TypeError is just a guess here. But
    # still we don't catch ValueError raised when an invalid hash_name is
    # passed (anything other than md5 or sha1).
    try:
        return joblib.hash(_dct, hash_name=method)
    except TypeError as ex:
        if raise_error:
            raise PsweepHashError(
                f"Error in hash calculation of: {dct}"
            ) from ex
        else:
            return np.nan


def add_hashes(params: Sequence[dict]) -> Sequence[dict]:
    """Calculate hash for each pset and set pset["_pset_hash"]. Return new
    params list.
    """
    new_params = []
    for pset in params:
        pset["_pset_hash"] = pset_hash(pset)
        new_params.append(pset)
    return new_params


def check_calc_dir(calc_dir: str, df: pd.DataFrame):
    """Check calc dir for consistency with database.

    Assuming dirs are named::

        <calc_dir>/<pset_id1>
        <calc_dir>/<pset_id2>
        ...

    check if we have matching dirs to ``_pset_id`` values in the
    database.
    """
    # fmt: off
    pset_ids_disk = set([
        m.group()
        for m in [
            re.match(r"(([0-9a-z]+)-){4}([0-9a-z]+)", x)
            for x in os.listdir(calc_dir)
        ] if m is not None])
    # fmt: on
    pset_ids_db = set(df._pset_id.values)
    return dict(
        db_not_disk=pset_ids_db - pset_ids_disk,
        disk_not_db=pset_ids_disk - pset_ids_db,
    )


def pickle_write(fn: str, obj):
    with open(fn, "wb") as fd:
        pickle.dump(obj, fd)


def pickle_read(fn: str):
    return pickle.load(open(fn, "rb"))


def logspace(
    start, stop, num=50, offset=0, log_func: Callable = np.log10, **kwds
):
    """
    Like ``numpy.logspace`` but `start` and `stop` are not exponents but the
    actual bounds.

    Control the strength of the log-scale by `offset`, where we use
    ``np.logspace(np.log10(start + offset), np.log10(stop + offset)) -
    offset``. `offset=0` is equal to ``np.logspace(np.log10(start),
    np.log10(stop))``. Higher values result in more evenly spaced points.

    Parameters
    ----------
    start, stop, num, **kwds :
        same as in ``np.logspace``
    offset :
        Control strength of log-scale.
    log_func :
        Must match `base`. Default is ``base=10`` as in ``np.logspace`` and so
        ``log_func=np.log10``. If you want a different `base`, also provide a
        matching `log_func`.

    Examples
    --------
    Effect of different `offset` values:

    >>> from matplotlib import pyplot as plt
    >>> from psweep import logspace
    >>> import numpy as np

    >>> for ii,off in enumerate([1e-16,1e-3, 1,2,3]):
    ...     x=logspace(0, 2, 20, offset=off)
    ...     plt.plot(x, np.ones_like(x)*ii, "o-", label=f"{off}")

    >>> plt.legend()
    """
    base = kwds.pop("base", 10.0)
    # fmt: off
    assert np.allclose(log_func(base), 1.0), f"log_func and {base=} don't match"
    # fmt: on
    return (
        np.logspace(
            log_func(start + offset),
            log_func(stop + offset),
            num=num,
            base=base,
            **kwds,
        )
        - offset
    )


def intspace(*args, dtype=np.int64, **kwds):
    """Like ``np.linspace`` but round to integers.

    The length of the returned array may be lower than specified by `num` if
    rounding to ints results in duplicates.

    Parameters
    ----------
    *args, **kwds :
        Same as ``np.linspace``
    """
    assert "dtype" not in kwds, "Got 'dtype' multiple times."
    return np.unique(np.round(np.linspace(*args, **kwds)).astype(dtype))


# -----------------------------------------------------------------------------
# git
# -----------------------------------------------------------------------------


def git_clean():
    return system("git status --porcelain").stdout.decode() == ""


def in_git_repo():
    # fmt: off
    return subprocess.run(
        "git status",
        check=False,
        shell=True,
        capture_output=True,
        ).returncode == 0
    # fmt: on


def git_enter(use_git: bool, always_commit=False):
    if use_git:
        path = os.path.basename(fullpath(os.curdir))
        if not in_git_repo():
            if always_commit:
                system(
                    f"git init; {GIT_ADD_ALL}; git commit -m 'psweep: {path}: init'"
                )
            else:
                raise Exception("no git repo here, create one first")
        if not git_clean():
            if always_commit:
                print("dirty repo, adding all changes")
                system(
                    f"{GIT_ADD_ALL}; git commit -m 'psweep: {path}: local changes'"
                )
            else:
                raise Exception("dirty repo, commit first")


def git_exit(use_git: bool, df: pd.DataFrame):
    if use_git and (not git_clean()):
        path = os.path.basename(fullpath(os.curdir))
        system(
            f"{GIT_ADD_ALL}; git commit -m 'psweep: {path}: run_id={df._run_id.values[-1]}'"
        )


# -----------------------------------------------------------------------------
# pandas
# -----------------------------------------------------------------------------


def df_to_json(df: pd.DataFrame, **kwds) -> str:
    """Like `df.to_json` but with defaults for orient, date_unit, date_format,
    double_precision.

    Parameters
    ----------
    df : DataFrame
    kwds :
        passed to :meth:`df.to_json`
    """
    defaults = dict(
        orient=PANDAS_DEFAULT_ORIENT,
        date_unit=PANDAS_TIME_UNIT,
        date_format="iso",
        double_precision=15,
    )
    for key, val in defaults.items():
        if key not in kwds.keys():
            kwds[key] = val
    return df.to_json(**kwds)


def df_write(fn: str, df: pd.DataFrame, fmt="pickle", **kwds) -> None:
    """Write DataFrame to disk.

    Parameters
    ----------
    fn :
        filename
    df : DataFrame
    fmt :
        ``{'pickle', 'json'}``
    kwds :
        passed to ``pickle.dump()`` or :func:`df_to_json`
    """
    _swapped_args_err_msg = (
        "Deprecated API in df_write(), use df_write(filename, df)"
    )
    if isinstance(fn, pd.DataFrame) or isinstance(df, str):
        if isinstance(fn, pd.DataFrame) and isinstance(df, str):
            warnings.warn(_swapped_args_err_msg, DeprecationWarning)
            tmp_fn = df
            df = fn
            fn = tmp_fn
        else:
            raise ValueError(_swapped_args_err_msg)

    makedirs(os.path.dirname(fn))
    if fmt == "pickle":
        with open(fn, "wb") as fd:
            pickle.dump(df, fd, **kwds)
    elif fmt == "json":
        df_to_json(df, path_or_buf=fn, **kwds)
    else:
        raise ValueError("unknown fmt: {}".format(fmt))


def df_read(fn: str, fmt="pickle", **kwds):
    """Read DataFrame from file `fn`. See :func:`df_write`."""
    if fmt == "pickle":
        with open(fn, "rb") as fd:
            return pickle.load(fd, **kwds)
    elif fmt == "json":
        orient = kwds.pop("orient", PANDAS_DEFAULT_ORIENT)
        return pd.io.json.read_json(
            fn, precise_float=True, orient=orient, **kwds
        )
    else:
        raise ValueError("unknown fmt: {}".format(fmt))


def df_print(
    df: pd.DataFrame,
    index: bool = False,
    special_cols: bool = False,
    cols: Sequence[str] = [],
    skip_cols: Sequence[str] = [],
):
    """Print DataFrame, by default without the index and special columns such
    as `_pset_id`.

    Similar logic as in `bin/psweep-db2table`, w/o tabulate support but more
    features (`skip_cols` for instance).

    Column names are always sorted, so the order of names in e.g. `cols`
    doesn't matter.

    Parameters
    ----------
    df : DataFrame
    index :
        include DataFrame index
    special_cols :
        include all special columns (`_pset_id` etc.)
    cols :
        explicit sequence of columns, overrides `special_cols` when special columns
        are specified
    skip_cols :
        skip those columns instead of selecting them (like `cols` would), use
        either this or `cols`; overrides `special_cols` when special columns
        are specified

    Examples
    --------
    >>> import pandas as pd
    >>> df=pd.DataFrame(dict(a=rand(3), b=rand(3), _c=rand(3)))

    >>> df
              a         b        _c
    0  0.373534  0.304302  0.161799
    1  0.698738  0.589642  0.557172
    2  0.343316  0.186595  0.822023

    >>> ps.df_print(df)
           a        b
    0.373534 0.304302
    0.698738 0.589642
    0.343316 0.186595

    >>> ps.df_print(df, special_cols=True)
           a        b       _c
    0.373534 0.304302 0.161799
    0.698738 0.589642 0.557172
    0.343316 0.186595 0.822023

    >>> ps.df_print(df, index=True)
              a        b
    0  0.373534 0.304302
    1  0.698738 0.589642
    2  0.343316 0.186595

    >>> ps.df_print(df, cols=["a"])
           a
    0.373534
    0.698738
    0.343316

    >>> ps.df_print(df, cols=["a"], special_cols=True)
           a       _c
    0.373534 0.161799
    0.698738 0.557172
    0.343316 0.822023

    >>> ps.df_print(df, cols=["a", "_c"])
           a       _c
    0.373534 0.161799
    0.698738 0.557172
    0.343316 0.822023

    >>> ps.df_print(df, skip_cols=["a"])
           b
    0.304302
    0.589642
    0.186595
    """
    _special_cols = set(x for x in df.columns if x.startswith("_"))
    if len(cols) > 0:
        if len(skip_cols) > 0:
            raise ValueError("Use either skip_cols or cols")
        disp_cols = set(cols) | (_special_cols if special_cols else set())
    else:
        disp_cols = set(df.columns) - (
            set() if special_cols else _special_cols
        )
        if len(skip_cols) > 0:
            disp_cols = disp_cols - set(skip_cols)
    disp_cols = list(disp_cols)
    disp_cols.sort()
    print(df[disp_cols].to_string(index=index))


def df_filter_conds(
    df: pd.DataFrame, conds: Sequence[Sequence[bool]]
) -> pd.DataFrame:
    """Filter DataFrame using bool arrays/Series/DataFrames in `conds`.

    Logical-and all bool sequences in `conds`. Same as

    >>> df[conds[0] & conds[1] & conds[2] & ...]

    but `conds` can be programmatically generated while the expression above
    would need to be changed by hand if `conds` changes.

    Parameters
    ----------
    df : DataFrame
    conds :
        sequence of bool masks, each of length `len(df)`

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
    cc = conds if hasattr(conds, "__len__") else list(conds)
    if len(cc) == 0:
        return df
    for ic, c in enumerate(cc):
        # fmt: off
        assert len(c) == len(df), \
            f"Condition at index {ic} has {len(c)=}, expect {len(df)=}"
        # fmt: on
    if len(cc) == 1:
        msk = cc[0]
    else:
        msk = np.logical_and.reduce(cc)
    return df[msk]


# -----------------------------------------------------------------------------
# building params
# -----------------------------------------------------------------------------


def plist(name: str, seq: Sequence[Any]):
    """Create a list of single-item dicts holding the parameter name and a
    value.

    >>> plist('a', [1,2,3])
    [{'a': 1}, {'a': 2}, {'a': 3}]
    """
    return [{name: entry} for entry in seq]


@itr
def merge_dicts(args: Sequence[dict]):
    """Start with an empty dict and update with each arg dict
    left-to-right."""
    dct = {}
    assert is_seq(args), f"input {args=} is no sequence"
    for entry in args:
        assert isinstance(entry, dict), f"{entry=} is no dict"
        dct.update(entry)
    return dct


def itr2params(loops: Iterator[Any]):
    """Transform the (possibly nested) result of a loop over plists (or
    whatever has been used to create psets) to a proper list of psets
    by flattening and merging dicts.

    Examples
    --------
    >>> a = ps.plist('a', [1,2])
    >>> b = ps.plist('b', [77,88])
    >>> c = ps.plist('c', ['const'])

    >>> # result of loops
    >>> list(itertools.product(a,b,c))
    [({'a': 1}, {'b': 77}, {'c': 'const'}),
     ({'a': 1}, {'b': 88}, {'c': 'const'}),
     ({'a': 2}, {'b': 77}, {'c': 'const'}),
     ({'a': 2}, {'b': 88}, {'c': 'const'})]

    >>> # flatten into list of psets
    >>> ps.itr2params(itertools.product(a,b,c))
    [{'a': 1, 'b': 77, 'c': 'const'},
     {'a': 1, 'b': 88, 'c': 'const'},
     {'a': 2, 'b': 77, 'c': 'const'},
     {'a': 2, 'b': 88, 'c': 'const'}]

    >>> # also more nested stuff is no problem
    >>> list(itertools.product(zip(a,b),c))
    [(({'a': 1}, {'b': 77}), {'c': 'const'}),
     (({'a': 2}, {'b': 88}), {'c': 'const'})]

    >>> ps.itr2params(itertools.product(zip(a,b),c))
    [{'a': 1, 'b': 77, 'c': 'const'},
     {'a': 2, 'b': 88, 'c': 'const'}]
    """
    ret = [merge_dicts(flatten(entry)) for entry in loops]
    lens = list(map(len, ret))
    assert (
        len(np.unique(lens)) == 1
    ), f"not all psets have same length {lens=}\n  {ret=}"
    return ret


@itr
def pgrid(plists):
    """Convenience function for the most common loop: nested loops with
    ``itertools.product``: ``ps.itr2params(itertools.product(a,b,c,...))``.

    Examples
    --------
    >>> a = ps.plist('a', [1,2])
    >>> b = ps.plist('b', [77,88])
    >>> c = ps.plist('c', ['const'])
    >>> # same as pgrid([a,b,c])
    >>> ps.pgrid(a,b,c)
    [{'a': 1, 'b': 77, 'c': 'const'},
     {'a': 1, 'b': 88, 'c': 'const'},
     {'a': 2, 'b': 77, 'c': 'const'},
     {'a': 2, 'b': 88, 'c': 'const'}]

    >>> ps.pgrid(zip(a,b),c)
    [{'a': 1, 'b': 77, 'c': 'const'},
     {'a': 2, 'b': 88, 'c': 'const'}]
    """
    assert is_seq(plists), f"input {plists=} is no sequence"
    return itr2params(itertools.product(*plists))


def filter_params_unique(params: Sequence[dict], **kwds) -> Sequence[dict]:
    """Reduce params to unique psets.

    Use pset["_pset_hash"]  if present, else calculate hash on the fly.

    Parameters
    ----------
    params
    kwds
        passed to :func:`pset_hash`
    """
    get_hash = lambda pset: pset.get("_pset_hash", pset_hash(pset, **kwds))
    msk = np.unique([get_hash(pset) for pset in params], return_index=True)[1]
    return [params[ii] for ii in np.sort(msk)]


def filter_same_hash(*args, **kwds):
    warnings.warn(
        "filter_same_hash() was renamed to filter_params_unique()",
        DeprecationWarning,
    )
    return filter_params_unique(*args, **kwds)


def filter_params_dup_hash(
    params: Sequence[dict], hashes: Sequence[str], **kwds
) -> Sequence[dict]:
    """Return params with psets whose hash is not in `hashes`.

    Use pset["_pset_hash"]  if present, else calculate hash on the fly.

    Parameters
    ----------
    params
    hashes
    kwds
        passed to :func:`pset_hash`
    """
    get_hash = lambda pset: pset.get("_pset_hash", pset_hash(pset, **kwds))
    return [pset for pset in params if not get_hash(pset) in hashes]


def stargrid(
    const: dict,
    vary: Sequence[dict],
    vary_labels: Sequence[str] = None,
    vary_label_col: str = "_vary",
    filter_dups=None,
    skip_dups=True,
) -> Sequence[dict]:
    """
    Helper to create a specific param sampling pattern.

    Vary params in a "star" pattern (and not a full pgrid) around constant
    values (middle of the "star").

    When doing that, duplicate psets can occur. By default try to filter them
    out (use filter_params_unique()) but ignore hash calculation errors and return
    non-reduced params in that case. If you want to fail at hash errors, use

    >>> filter_params_unique(stargrid(..., skip_dups=False), raise_error=True)

    Examples
    --------
    >>> from psweep import psweep as ps
    >>> const=dict(a=1, b=77, c=11)
    >>> a=ps.plist("a", [1,2,3,4])
    >>> b=ps.plist("b", [77,88,99])
    >>> c=ps.plist("c", [11,22,33,44])

    >>> ps.stargrid(const, vary=[a, b])
    [{'a': 1, 'b': 77, 'c': 11},
     {'a': 2, 'b': 77, 'c': 11},
     {'a': 3, 'b': 77, 'c': 11},
     {'a': 4, 'b': 77, 'c': 11},
     {'a': 1, 'b': 88, 'c': 11},
     {'a': 1, 'b': 99, 'c': 11}]

    >>> ps.stargrid(const, vary=[a, b], skip_dups=False)
    [{'a': 1, 'b': 77, 'c': 11},
     {'a': 2, 'b': 77, 'c': 11},
     {'a': 3, 'b': 77, 'c': 11},
     {'a': 4, 'b': 77, 'c': 11},
     {'a': 1, 'b': 77, 'c': 11},
     {'a': 1, 'b': 88, 'c': 11},
     {'a': 1, 'b': 99, 'c': 11}]

    >>> ps.stargrid(const, vary=[a, b], vary_labels=["a", "b"])
    [{'a': 1, 'b': 77, 'c': 11, '_vary': 'a'},
     {'a': 2, 'b': 77, 'c': 11, '_vary': 'a'},
     {'a': 3, 'b': 77, 'c': 11, '_vary': 'a'},
     {'a': 4, 'b': 77, 'c': 11, '_vary': 'a'},
     {'a': 1, 'b': 88, 'c': 11, '_vary': 'b'},
     {'a': 1, 'b': 99, 'c': 11, '_vary': 'b'}]

    >>> ps.stargrid(const, vary=[ps.itr2params(zip(a,c)),b], vary_labels=["a+c", "b"])
    [{'a': 1, 'b': 77, 'c': 11, '_vary': 'a+c'},
     {'a': 2, 'b': 77, 'c': 22, '_vary': 'a+c'},
     {'a': 3, 'b': 77, 'c': 33, '_vary': 'a+c'},
     {'a': 4, 'b': 77, 'c': 44, '_vary': 'a+c'},
     {'a': 1, 'b': 88, 'c': 11, '_vary': 'b'},
     {'a': 1, 'b': 99, 'c': 11, '_vary': 'b'}]
    """
    params = []
    if vary_labels is not None:
        assert len(vary_labels) == len(
            vary
        ), f"{vary_labels=} and {vary=} must have same length"
    for ii, plist in enumerate(vary):
        for dct in plist:
            if vary_labels is not None:
                label = {vary_label_col: vary_labels[ii]}
                _dct = merge_dicts(dct, label)
            else:
                _dct = dct
            params.append(merge_dicts(const, _dct))

    if filter_dups is not None:
        skip_dups = filter_dups
        warnings.warn(
            "filter_dups was renamed to skip_dups", DeprecationWarning
        )
    if skip_dups:
        try:
            return filter_params_unique(
                params, raise_error=True, skip_special_cols=True
            )
        except PsweepHashError:
            return params
    else:
        return params


# -----------------------------------------------------------------------------
# run study
# -----------------------------------------------------------------------------

# tmpsave: That's cool, but when running in parallel, we loose the ability to
# store the whole state of the study calculated thus far. For that we would
# need an extra thread that periodically checks for or -- even better -- gets
# informed by workers about finished work and collects the so-far written temp
# results into a global df -- maybe useful for monitoring progress.


def worker_wrapper(
    pset: dict,
    worker: Callable,
    tmpsave: bool = False,
    verbose: Union[bool, Sequence[str]] = None,
    run_id: str = None,
    calc_dir: str = None,
    simulate: bool = False,
    pset_seq=np.nan,
    run_seq: int = None,
):
    """
    Add special fields to pset. Call worker on exactly one pset. Return
    DataFrame row built from ``pset.update(worker(pset))``.
    """
    assert run_id is not None
    assert calc_dir is not None
    pset_id = str(uuid.uuid4())
    _pset = copy.deepcopy(pset)
    time_start = pd.Timestamp(time.time(), unit=PANDAS_TIME_UNIT)
    update = {
        "_run_id": run_id,
        "_pset_id": pset_id,
        "_calc_dir": calc_dir,
        "_time_utc": time_start,
        "_pset_seq": pset_seq,
        "_run_seq": run_seq,
    }
    _pset.update(update)
    if verbose is not None:
        df_row_print = pd.DataFrame([_pset], index=[time_start])
        if isinstance(verbose, bool) and verbose:
            df_print(df_row_print, index=True)
        elif is_seq(verbose):
            df_print(df_row_print[verbose], index=True)
    t0 = time.time()
    if not simulate:
        _pset.update(worker(_pset))
    _pset["_pset_runtime"] = time.time() - t0
    df_row = pd.DataFrame([_pset])
    if tmpsave:
        fn = pj(calc_dir, "tmpsave", run_id, pset_id + ".pk")
        df_write(fn, df_row)
    return df_row


def run_local(
    worker: Callable,
    params: Sequence[dict],
    df: pd.DataFrame = None,
    poolsize: int = None,
    save=True,
    tmpsave=False,
    verbose: Union[bool, Sequence[str]] = False,
    calc_dir="calc",
    simulate=False,
    database_dir: str = None,
    database_basename="database.pk",
    backup=False,
    git=False,
    skip_dups=False,
) -> pd.DataFrame:
    """
    Call `worker` for each `pset` in `params`. Populate a DataFrame with rows
    from each call ``worker(pset)``.

    Parameters
    ----------
    worker : Callable
        must accept one parameter: `pset` (a dict ``{'a': 1, 'b': 'foo',
        ...}``), return either an update to `pset` or a new dict, result will
        be processes as ``pset.update(worker(pset))``
    params : seq of dicts
        each dict is a pset ``{'a': 1, 'b': 'foo', ...}``
    df : DataFrame
        append rows to this DataFrame, if None then either create new one or
        read existing database file from disk if found
    poolsize : {None, int}
        * None : use serial execution
        * int : use multiprocessing.Pool (even for ``poolsize=1``)
    save : bool
        save final DataFrame to ``<database_dir>/<database_basename>`` (pickle
        format only), default: "calc/database.pk", see also `database_dir`,
        `calc_dir` and `database_basename`
    tmpsave : bool
        save results from each `pset` in `params` (the current DataFrame row) to
        ``<calc_dir>/tmpsave/<run_id>/<pset_id>.pk`` (pickle format only)
    verbose : {bool, sequence of str}
        * bool : print the current DataFrame row
        * sequence : list of DataFrame column names, print the row but only
          those columns
    calc_dir : str
        Dir where calculation artifacts can be saved if needed, such as dirs
        per pset ``<calc_dir>/<pset>``. Will be added to the database in
        ``_calc_dir`` field.
    simulate : bool
        run everything in ``<calc_dir>.simulate``, don't call `worker`, i.e. save
        what the run would create, but without the results from `worker`,
        useful to check if `params` are correct before starting a production run
    database_dir : str
        Path for the database. Default is ``<calc_dir>``.
    database_basename : str
        ``<database_dir>/<database_basename>``, default: "database.pk"
    backup : bool
        Make backup of ``<calc_dir>`` to ``<calc_dir>.bak_<timestamp>_run_id_<_run_id>``
    git : bool
        Use ``git`` to commit all files written and changed by the current run
        (``_run_id``). Make sure to create a ``.gitignore`` manually before if
        needed.
    skip_dups : bool
        Skip psets whose hash is already present in `df`. Useful when repeating
        (parts of) a study.
    """

    database_dir = calc_dir if database_dir is None else database_dir

    git_enter(git)

    if simulate:
        calc_dir_sim = calc_dir + ".simulate"
        if os.path.exists(calc_dir_sim):
            shutil.rmtree(calc_dir_sim)
        makedirs(calc_dir_sim)
        old_db = pj(database_dir, database_basename)
        if os.path.exists(old_db):
            shutil.copy(old_db, pj(calc_dir_sim, database_basename))
        else:
            warnings.warn(
                f"simulate: {old_db} not found, will create new db in "
                f"{calc_dir_sim}"
            )
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

    if backup and len(df.index) > 0:
        stamp = df._time_utc.max().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        dst = f"{calc_dir}.bak_{stamp}_run_id_{df._run_id.values[-1]}"
        assert not os.path.exists(dst), (
            "backup destination {dst} exists, seems like there has been no new "
            "data in {calc_dir} since the last backup".format(
                dst=dst, calc_dir=calc_dir
            )
        )
        shutil.copytree(calc_dir, dst)

    params = add_hashes(params)

    if skip_dups and len(df) > 0:
        params = filter_params_dup_hash(params, df._pset_hash.values)

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
        df = pd.concat((df, df_row), sort=False, ignore_index=True)

    if save:
        df_write(database_fn, df)

    git_exit(git, df)

    return df


# -----------------------------------------------------------------------------
# HPC cluster batch runs
# -----------------------------------------------------------------------------


class Machine:
    def __init__(self, machine_dir: str, jobscript_name: str = "jobscript"):
        """
        Expected templates layout::

            templates/machines/<name>/info.yaml
            ^^^^^^^^^^^^^^^^^^^^^^^^^------------- machine_dir
            templates/machines/<name>/jobscript
                                      ^^^^^^^^^--- template.basename


        """
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
        try:
            return string.Template(file_read(self.filename)).substitute(pset)
        except:
            print(f"Failed to fill template: {self.filename}", file=sys.stderr)
            raise


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


# If we ever add a "simulate" kwd here: don't pass that thru to run_local() b/c
# there this prevents worker() from being executed, but that's what we always
# want here since it writes only input files. Instead, just set calc_dir =
# calc_dir_sim and copy the database as in run_local() and go. Don't copy the
# run_*.sh scripts b/c they are generated afresh anyway.
#
def prep_batch(
    params: Sequence[dict],
    calc_dir: str = "calc",
    calc_templ_dir: str = "templates/calc",
    machine_templ_dir: str = "templates/machines",
    git: bool = False,
    backup: bool = False,
    write_pset: bool = False,
    skip_dups: bool = False,
) -> pd.DataFrame:

    """
    Write files based on templates.
    """
    git_enter(git)

    calc_templates = gather_calc_templates(calc_templ_dir)
    machines = gather_machines(machine_templ_dir)
    templates = calc_templates + [m.template for m in machines]

    def worker(pset):
        for template in templates:
            file_write(
                pj(calc_dir, pset["_pset_id"], template.targetname),
                template.fill(pset),
            )
            if write_pset:
                pickle_write(pj(calc_dir, pset["_pset_id"], "pset.pk"), pset)
        return {}

    df = run_local(
        worker,
        params,
        calc_dir=calc_dir,
        backup=backup,
        skip_dups=skip_dups,
        git=False,
    )

    msk_latest = df._run_seq == df._run_seq.values.max()
    msk_old = df._run_seq < df._run_seq.values.max()
    for machine in machines:
        txt = ""
        for pfx, msk in [("# ", msk_old), ("", msk_latest)]:
            if msk.any():
                txt += "\n"
            txt += "\n".join(
                f"{pfx}cd $here/{pset_id}; {machine.subcmd} {machine.template.targetname}  # run_seq={run_seq} pset_seq={pset_seq}"
                for pset_id, pset_seq, run_seq in zip(
                    df[msk]._pset_id.values,
                    df[msk]._pset_seq.values,
                    df[msk]._run_seq.values,
                )
            )
        file_write(
            f"{calc_dir}/run_{machine.name}.sh",
            f"#!/bin/sh\n\nhere=$(readlink -f $(dirname $0))\n{txt}\n",
        )

    git_exit(git, df)

    return df
