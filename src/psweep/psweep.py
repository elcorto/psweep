from functools import partial, wraps
from io import IOBase, StringIO
from typing import Any, Sequence, Callable, Iterator
import copy
from contextlib import redirect_stdout, redirect_stderr
import itertools
import multiprocessing as mp
import os
import pickle
import platform
import re
import shutil
import string
import subprocess
import sys
import time
import uuid
import warnings
import json

import joblib
import numpy as np
import pandas as pd
import yaml
import jinja2


pj = os.path.join

# defaults, globals
PANDAS_DEFAULT_ORIENT = "records"
PANDAS_TIME_UNIT = "s"
PSET_HASH_ALG = "sha1"
GIT_ADD_ALL = "git add -A -v"
CALC_DIR = "calc"
DATABASE_BASENAME = "database.pk"
FILL_VALUE = pd.NA

# Make DeprecationWarning visible to users by default.
warnings.simplefilter("default")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def system(cmd: str, **kwds) -> subprocess.CompletedProcess:
    """
    Call shell command.

    Parameters
    ----------
    cmd
        shell command
    kwds
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
    """Wrap `func` to allow passing args not as sequence.

    Assuming ``func()`` requires a sequence as input: ``func([a,b,c])``, allow
    passing ``func(a,b,c)``.
    """

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
        try:
            fd.write(txt)
        except UnicodeEncodeError:
            fd.write(txt.encode("ascii", errors="xmlcharrefreplace").decode())


def file_read(fn: str):
    with open(fn, "r") as fd:
        return fd.read()


def pickle_write(fn: str, obj):
    makedirs(os.path.dirname(fn))
    with open(fn, "wb") as fd:
        pickle.dump(obj, fd)


def pickle_read(fn: str):
    with open(fn, "rb") as fd:
        return pickle.load(fd)


def json_write(fn: str, obj):
    file_write(fn, json.dumps(obj, indent=2))


def json_read(fn: str):
    with open(fn, "r") as fd:
        return json.load(fd)


class PsweepHashError(TypeError):
    pass


def _get_col_filter(skip_prefix_cols=True, skip_postfix_cols=True):
    """Implement the package-wide prefix/postfix default convention.

    By default, we ignore prefix fields (book-keeping) and postfix fields
    (results).
    """
    if skip_prefix_cols and skip_postfix_cols:
        return lambda key: not (key.startswith("_") or key.endswith("_"))
    elif skip_prefix_cols:
        return lambda key: not key.startswith("_")
    elif skip_postfix_cols:
        return lambda key: not key.endswith("_")
    else:
        return lambda key: True


def pset_hash(
    dct: dict,
    method=PSET_HASH_ALG,
    raise_error=True,
    **kwds,
):
    """Reproducible hash of a dict for usage in database (hash of a `pset`).

    We implement the convention to ignore prefix fields (book-keeping) and
    postfix fields (results). You can pass `skip_prefix_cols` /
    `skip_postfix_cols` to change that (see :func:`_get_col_filter`).
    """

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
    # even though for our purpose, we'd consider the two instances of ``Foo``
    # to be the same.
    #
    # The same observations have been also made elsewhere [1,2]. Esp. [2]
    # points to [3] which in turn mentions joblib.hashing.hash(). It's code
    # shows how complex the problem is, but so far this is our best bet.
    #
    # [1] https://death.andgravity.com/stable-hashing
    # [2] https://ourpython.com/python/deterministic-recursive-hashing-in-python
    # [3] https://stackoverflow.com/a/52175075
    assert isinstance(dct, dict), f"{dct=} is not a dict but {type(dct)=}"
    filt = _get_col_filter(**kwds)
    _dct = {key: val for key, val in dct.items() if filt(key)}
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


def logspace(
    start, stop, num=50, offset=0, log_func: Callable = np.log10, **kwds
):
    """
    Like ``numpy.logspace`` but

    * `start` and `stop` are not exponents but the actual bounds
    * tuneable log scale strength

    Control the strength of the log scale by `offset`, where we use by default
    ``log_func=np.log10`` and ``base=10`` and return
    ``np.logspace(np.log10(start + offset), np.log10(stop + offset)) -
    offset``. `offset=0` is equal to ``np.logspace(np.log10(start),
    np.log10(stop))``. Higher `offset` values result in more evenly spaced
    points.

    Parameters
    ----------
    start, stop, num, **kwds :
        same as in ``np.logspace``
    offset :
        Control strength of log scale.
    log_func :
        Must match `base` (pass that as part of `**kwds`). Default is
        ``base=10`` as in ``np.logspace`` and so ``log_func=np.log10``. If you
        want a different `base`, also provide a matching `log_func`, e.g.
        ``base=e, log_func=np.log``.

    Examples
    --------
    Effect of different `offset` values:

    >>> from matplotlib import pyplot as plt
    >>> from psweep import logspace
    >>> import numpy as np
    >>> for ii, offset in enumerate([1e-16,1e-3, 1,2,3]):
    ...     x=logspace(0, 2, 20, offset=offset)
    ...     plt.plot(x, np.ones_like(x)*ii, "o-", label=f"{offset=}")
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
    *args, **kwds
        Same as ``np.linspace``
    """
    assert "dtype" not in kwds, "Got 'dtype' multiple times."
    return np.unique(np.round(np.linspace(*args, **kwds)).astype(dtype))


def get_uuid(retry=10, existing: Sequence = []) -> str:
    ret = str(uuid.uuid4())
    while ret in existing:
        ret = str(uuid.uuid4())
        retry -= 1
        if retry == 0:
            raise Exception(
                f"Failed to generate UUID after {retry} attempts. "
                f"Existing UUIDs: {existing}"
            )
    return ret


def get_many_uuids(
    num: int, retry=10, existing: Sequence = []
) -> Sequence[str]:
    generate = lambda: set(str(uuid.uuid4()) for _ in range(num))
    ret = generate()
    set_existing = set(existing)
    while (len(ret) < num) or (len(ret & set_existing) > 0):
        ret = generate()
        retry -= 1
        if retry == 0:
            raise Exception(
                f"Failed to generate {num} UUIDs after {retry} attempts. "
                f"Existing UUIDs: {existing}"
            )
    return list(ret)


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
    df
        DataFrame to convert
    kwds
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
    fn
        filename
    df
        DataFrame to write
    fmt
        ``{'pickle', 'json'}``
    kwds
        passed to ``pickle.dump()`` or :func:`df_to_json`
    """
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
    special_cols=None,
    prefix_cols: bool = False,
    cols: Sequence[str] = [],
    skip_cols: Sequence[str] = [],
):
    """Print DataFrame, by default without the index and prefix columns such
    as `_pset_id`.

    Similar logic as in `bin/psweep-db2table`, w/o tabulate support but more
    features (`skip_cols` for instance).

    Column names are always sorted, so the order of names in e.g. `cols`
    doesn't matter.

    Parameters
    ----------
    df
    index
        include DataFrame index
    prefix_cols
        include all prefix columns (`_pset_id` etc.), we don't support skipping
        user-added postfix columns (e.g. `result_`)
    cols
        explicit sequence of columns, overrides `prefix_cols` when prefix columns
        are specified
    skip_cols
        skip those columns instead of selecting them (like `cols` would), use
        either this or `cols`; overrides `prefix_cols` when prefix columns
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

    >>> ps.df_print(df, prefix_cols=True)
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

    >>> ps.df_print(df, cols=["a"], prefix_cols=True)
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
    if special_cols is not None:
        warnings.warn(
            "special_cols is deprecated, use prefix_cols",
            DeprecationWarning,
        )
        prefix_cols = special_cols

    _prefix_cols = set(x for x in df.columns if x.startswith("_"))
    if len(cols) > 0:
        if len(skip_cols) > 0:
            raise ValueError("Use either skip_cols or cols")
        disp_cols = set(cols) | (_prefix_cols if prefix_cols else set())
    else:
        disp_cols = set(df.columns) - (set() if prefix_cols else _prefix_cols)
        if len(skip_cols) > 0:
            disp_cols = disp_cols - set(skip_cols)
    disp_cols = list(disp_cols)
    disp_cols.sort()
    print(df[disp_cols].to_string(index=index))


def df_filter_conds(
    df: pd.DataFrame,
    conds: Sequence[Sequence[bool]],
    op: str = "and",
) -> pd.DataFrame:
    """Filter DataFrame using bool arrays/Series/DataFrames in `conds`.

    Fuse all bool sequences in `conds` using `op`. For instance, if
    ``op="and"``, then we logical-and them, which is equal to

    >>> df[conds[0] & conds[1] & conds[2] & ...]

    but `conds` can be programmatically generated while the expression above
    would need to be changed by hand if `conds` changes.

    Parameters
    ----------
    df
        DataFrame
    conds
        Sequence of bool masks, each of length `len(df)`.
    op
        Bool operator, used as ``numpy.logical_{op}``, e.g. "and", "or",
        "xor".

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
        assert op in (op_allowed := ["and", "or", "xor"]), (
            f"{op=} not one of {op_allowed}"
        )
        msk = getattr(np, f"logical_{op}").reduce(cc)
    return df[msk]


def df_update_pset_hash(df: pd.DataFrame, copy: bool = False) -> pd.DataFrame:
    """Add or update ``_pset_hash`` column."""
    df_out = df.copy() if copy else df
    for idx, row_dct in enumerate(df_extract_dicts(df)):
        df_out.at[idx, "_pset_hash"] = pset_hash(row_dct)
    return df_out


def df_update_pset_cols(
    df: pd.DataFrame,
    pset_cols: Sequence["str"] | set[str],
    fill_value=FILL_VALUE,
    copy: bool = False,
) -> pd.DataFrame:
    """
    Make sure that `df` has at least `pset_cols` columns. If not, add
    missing columns, fill with `fill_value`. Always refresh ``_pset_hash``.
    """
    df_out = df.copy() if copy else df
    old_pset_cols = set(filter_cols(df.columns, kind="pset"))
    new_pset_cols = set(pset_cols)
    assert len(new_pset_cols) >= len(old_pset_cols), (
        f"{old_pset_cols=} {new_pset_cols=}: "
        f"{len(new_pset_cols)=} < {len(old_pset_cols)=}"
    )
    if old_pset_cols != new_pset_cols:
        for col in new_pset_cols - old_pset_cols:
            df_out[col] = fill_value
    return df_update_pset_hash(df_out, copy=False)


def df_extract_dicts(
    df: pd.DataFrame, py_types: bool = False
) -> Sequence[dict]:
    """
    Convert `df`'s rows to dicts.

    Parameters
    ----------
    df
    py_types
        If True, let Pandas (``Series.to_dict()``) decide types. It tries
        to return Python native types (e.g. it converts ``pd.NA`` to ``None``.)
        Else, try to preserve types as they are in `df`.
    """
    if py_types:
        return [ser.to_dict() for (_, ser) in df.iterrows()]
    else:
        # itertuples() preserves type, while iterrows() doesn't (see iterrows()
        # docstring). But the default (itertuples(..., name='Pandas')) returns
        # namedtuples -- the worst data structure ever invented -- which can't
        # handle field names starting with "_", it will just rename them!
        #
        #    >>> df
        #       _a  b_
        #    0   1  77
        #    1   1  88
        #    2   2  77
        #    3   2  88
        #
        #    >>> list(df.itertuples(index=False))
        #    [Pandas(_0=1, b_=77),
        #     Pandas(_0=1, b_=88),
        #     Pandas(_0=2, b_=77),
        #     Pandas(_0=2, b_=88)]
        #
        # With name=None we get plain types and zip() the column names back in.
        return [
            dict(zip(df.columns, row))
            for row in df.itertuples(index=False, name=None)
        ]


def df_extract_params(
    df: pd.DataFrame, py_types: bool = False
) -> Sequence[dict]:
    """Extract `params` (list of psets) from `df`.

    Same as :func:`df_extract_dicts`, but limit columns to ``kind="pset"`` (see
    :func:`filter_cols`). This will reproduce the `params` fed to :func:`run`
    when following the prefix/postfix convention (see :func:`_get_col_filter`),
    meaning that the pset hashes will be the same.

    Parameters
    ----------
    df
    py_types
        See :func:`df_extract_dicts`

    Examples
    --------
    >>> import psweep as ps
    >>> from numpy.random import rand
    >>> params=ps.pgrid(ps.plist("a", [1,2,3]), ps.plist("b", [77,88]))
    >>> params
    [{'a': 1, 'b': 77},
     {'a': 1, 'b': 88},
     {'a': 2, 'b': 77},
     {'a': 2, 'b': 88},
     {'a': 3, 'b': 77},
     {'a': 3, 'b': 88}]

    >>> df=ps.run(func=lambda pset: dict(result_=rand()), params=params, save=False)
    >>> ps.df_extract_params(df)
    [{'a': 1, 'b': 77},
     {'a': 1, 'b': 88},
     {'a': 2, 'b': 77},
     {'a': 2, 'b': 88},
     {'a': 3, 'b': 77},
     {'a': 3, 'b': 88}]
    """
    return df_extract_dicts(
        df[filter_cols(df.columns, kind="pset")], py_types=py_types
    )


def df_extract_pset(
    df: pd.DataFrame, pset_id: str, py_types: bool = False
) -> dict:
    """Extract a single pset dict for `pset_id` from `df`.

    Parameters
    ----------
    df
    pset_id
    py_types
        See :func:`df_extract_dicts`
    """
    df_sel = df.loc[
        df._pset_id == pset_id, filter_cols(df.columns, kind="pset")
    ]
    assert len(df_sel) == 1, (
        "Selection is not unique, you have duplicate pset_ids!"
    )
    ser = df_sel.iloc[0]
    if py_types:
        return ser.to_dict()
    else:
        # Series.to_dict() casts pd.NA to None. The trick below preserves types
        # (at least NA).
        return dict(zip(ser.keys(), ser.to_list()))


def df_ensure_dtypes(df, fill_value=FILL_VALUE):
    """Make sure that `df`'s dtype is ``object``. Convert any ``pd.isna()``
    values to `fill_value`.

    This is part of our attempt to prevent pandas from doing type inference and
    conversion.
    """

    # pd.concat() can convert pd.NA back to NaN, fix that. If the element is
    # not "scalar", then give up and don't touch it.
    def apply_func(x):
        isna_result = pd.isna(x)
        if hasattr(isna_result, "__len__"):
            return x
        else:
            return fill_value if isna_result else x

    for col in df.columns:
        df[col] = df[col].apply(apply_func)

    # apply() changes dtype=object back to native types again, revert to object
    return df.astype(object)


def filter_cols(cols: Sequence[str], kind: str = "pset") -> Sequence[str]:
    if kind == "pset":
        filt = _get_col_filter(skip_prefix_cols=True, skip_postfix_cols=True)
    elif kind in ["pre", "prefix"]:
        filt = lambda key: key.startswith("_")
    elif kind in ["post", "postfix"]:
        filt = lambda key: key.endswith("_")
    else:
        raise ValueError(f"{kind=} not supported")
    return list(filter(filt, cols))


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
    assert len(np.unique(lens)) == 1, (
        f"not all psets have same length {lens=}\n  {ret=}"
    )
    return ret


@itr
def pgrid(plists: Sequence[Sequence[dict]]) -> Sequence[dict]:
    """Convenience function for the most common loop: nested loops with
    ``itertools.product``: ``ps.itr2params(itertools.product(a,b,c,...))``.

    Parameters
    ----------
    plists
        List of :func:`plist()` results. If more than one, you can also provide
        plists as args, so ``pgrid(a,b,c)`` instead of ``pgrid([a,b,c])``.

    Notes
    -----
    For a single plist arg, you have to use ``pgrid([a])``. ``pgrid(a)`` won't
    work. However, this edge case (passing one plist to pgrid) is not super
    useful, since

    >>> a=ps.plist("a", [1,2,3])
    >>> a
    [{'a': 1}, {'a': 2}, {'a': 3}]
    >>> ps.pgrid([a])
    [{'a': 1}, {'a': 2}, {'a': 3}]

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


def filter_params_unique(params: Sequence[dict]) -> Sequence[dict]:
    """Reduce params to unique psets.

    Use ``pset["_pset_hash"]`` if present, else calculate hash on the fly.

    Parameters
    ----------
    params
    """
    get_hash = lambda pset: pset.get("_pset_hash", pset_hash(pset))
    msk = np.unique([get_hash(pset) for pset in params], return_index=True)[1]
    return [params[ii] for ii in np.sort(msk)]


def filter_params_dup_hash(
    params: Sequence[dict], hashes: Sequence[str]
) -> Sequence[dict]:
    """Return params with psets whose hash is not in `hashes`.

    Use ``pset["_pset_hash"]`` if present, else calculate hash on the fly.

    Parameters
    ----------
    params
    hashes
    """
    get_hash = lambda pset: pset.get("_pset_hash", pset_hash(pset))
    return [pset for pset in params if get_hash(pset) not in hashes]


def stargrid(
    const: dict,
    vary: Sequence[Sequence[dict]],
    vary_labels: Sequence[str] = None,
    vary_label_col: str = "_vary",
    skip_dups=True,
) -> Sequence[dict]:
    """
    Helper to create a specific param sampling pattern.

    Vary params in a "star" pattern (and not a full pgrid) around constant
    values (middle of the "star").

    Parameters
    ----------
    const
        constant params
    vary
        list of plists
    vary_labels
        database col names for parameters in `vary`
    skip_dups
        filter duplicate psets (see Notes below)

    Notes
    -----
    `skip_dups`: When creating a star pattern, duplicate psets can occur. By
    default try to filter them out (using :func:`filter_params_unique`) but
    ignore hash calculation errors and return non-reduced params in that case.

    Examples
    --------
    >>> import psweep as ps
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

    >>> ps.stargrid(const, vary=[ps.pgrid([zip(a,c)]),b], vary_labels=["a+c", "b"])
    [{'a': 1, 'b': 77, 'c': 11, '_vary': 'a+c'},
     {'a': 2, 'b': 77, 'c': 22, '_vary': 'a+c'},
     {'a': 3, 'b': 77, 'c': 33, '_vary': 'a+c'},
     {'a': 4, 'b': 77, 'c': 44, '_vary': 'a+c'},
     {'a': 1, 'b': 88, 'c': 11, '_vary': 'b'},
     {'a': 1, 'b': 99, 'c': 11, '_vary': 'b'}]
    """
    params = []
    if vary_labels is not None:
        assert len(vary_labels) == len(vary), (
            f"{vary_labels=} and {vary=} must have same length"
        )
    for ii, plist in enumerate(vary):
        for dct in plist:
            if vary_labels is not None:
                label = {vary_label_col: vary_labels[ii]}
                _dct = merge_dicts(dct, label)
            else:
                _dct = dct
            params.append(merge_dicts(const, _dct))

    if skip_dups:
        try:
            return filter_params_unique(params)
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


def func_wrapper(
    pset: dict,
    func: Callable,
    *,
    tmpsave: bool = False,
    verbose: bool | Sequence[str] = False,
    simulate: bool = False,
) -> dict:
    """
    Add those prefix fields (e.g. `_time_utc`) to `pset` which can be
    determined at call time.

    Call `func` on exactly one pset. Return updated pset built from
    ``pset.update(func(pset))``. Do verbose printing.
    """
    assert "_pset_id" in pset
    assert "_run_id" in pset
    assert "_calc_dir" in pset

    time_start = pd.Timestamp(time.time(), unit=PANDAS_TIME_UNIT)
    pset.update(_time_utc=time_start, _exec_host=platform.node())
    if verbose:
        df_row_print = pd.DataFrame([pset], index=[time_start])
        if isinstance(verbose, bool) and verbose:
            df_print(df_row_print, index=True)
        elif is_seq(verbose):
            df_print(df_row_print, index=True, cols=verbose)
        else:
            raise ValueError(f"Type of {verbose=} not understood.")
    t0 = time.time()
    if not simulate:
        pset.update(func(pset))
    pset["_pset_runtime"] = time.time() - t0
    if tmpsave:
        fn = pj(
            pset["_calc_dir"],
            "tmpsave",
            pset["_run_id"],
            pset["_pset_id"] + ".pk",
        )
        pickle_write(fn, pset)
    return pset


def capture_logs_wrapper(
    pset: dict,
    func: Callable,
    capture_logs: str,
    db_field: str = "_logs",
) -> dict:
    """Capture and redirect stdout and stderr produced in func().

    Note the limitations mentioned in [1]:

        Note that the global side effect on sys.stdout means that this context
        manager is not suitable for use in library code and most threaded
        applications. It also has no effect on the output of subprocesses.
        However, it is still a useful approach for many utility scripts.

    So if users rely on playing with sys.stdout/stderr in func(), then they
    should not use this feature and take care of logging themselves.

    [1] https://docs.python.org/3/library/contextlib.html#contextlib.redirect_stdout
    """
    fn = f"{pset['_calc_dir']}/{pset['_pset_id']}/logs.txt"
    if capture_logs == "file":
        makedirs(os.path.dirname(fn))
        with open(fn, "w") as fd, redirect_stdout(fd), redirect_stderr(fd):
            return func(pset)
    elif capture_logs in ["db", "db+file"]:
        with StringIO() as fd:
            with redirect_stdout(fd), redirect_stderr(fd):
                ret = func(pset)
            txt = fd.getvalue()
        ret[db_field] = txt
        if capture_logs == "db+file":
            file_write(fn, txt)
        return ret
    else:
        raise ValueError(f"Illegal value {capture_logs=}")


def _setup_simulate_dir(*, calc_dir, database_basename):
    calc_dir_sim = calc_dir + ".simulate"
    if os.path.exists(calc_dir_sim):
        shutil.rmtree(calc_dir_sim)
    makedirs(calc_dir_sim)
    old_db = pj(calc_dir, database_basename)
    if os.path.exists(old_db):
        shutil.copy(old_db, pj(calc_dir_sim, database_basename))
    else:
        warnings.warn(
            f"simulate: {old_db} not found, will create new db in "
            f"{calc_dir_sim}"
        )
    return calc_dir_sim


def run(
    func: Callable,
    params: Sequence[dict],
    df: pd.DataFrame = None,
    poolsize: int = None,
    dask_client=None,
    save: bool = True,
    tmpsave: bool = False,
    verbose: bool | Sequence[str] = False,
    calc_dir: str = CALC_DIR,
    simulate: bool = False,
    database_basename: str = DATABASE_BASENAME,
    backup: bool = False,
    git: bool = False,
    skip_dups: bool = False,
    capture_logs: str = None,
    fill_value=FILL_VALUE,
) -> pd.DataFrame:
    """
    Call `func` for each `pset` in `params`. Populate a DataFrame with rows
    from each call ``func(pset)``.

    Parameters
    ----------
    func
        must accept one parameter: `pset` (a dict ``{'a': 1, 'b': 'foo',
        ...}``), return either an update to `pset` or a new dict, result will
        be processes as ``pset.update(func(pset))``
    params
        each dict is a pset ``{'a': 1, 'b': 'foo', ...}``
    df
        append rows to this DataFrame, if None then either create new one or
        read existing database file from disk if found
    poolsize
        * None : use serial execution
        * int : use multiprocessing.Pool (even for ``poolsize=1``)
    dask_client
        A dask client. Use this or ``poolsize``.
    save
        save final ``DataFrame`` to ``<calc_dir>/<database_basename>`` (pickle
        format only), default: "calc/database.pk", see also `calc_dir` and
        `database_basename`
    tmpsave
        save the result dict from each ``pset.update(func(pset))`` from each
        `pset` to ``<calc_dir>/tmpsave/<run_id>/<pset_id>.pk`` (pickle format
        only), the data is a dict, not a DataFrame row
    verbose
        * bool : print the current DataFrame row
        * sequence : list of DataFrame column names, print the row but only
          those columns
    calc_dir
        Dir where calculation artifacts can be saved if needed, such as dirs
        per pset ``<calc_dir>/<pset_id>``. Will be added to the database in
        ``_calc_dir`` field.
    simulate
        run everything in ``<calc_dir>.simulate``, don't call `func`, i.e. save
        what the run would create, but without the results from `func`,
        useful to check if `params` are correct before starting a production run
    database_basename
        ``<calc_dir>/<database_basename>``, default: "database.pk"
    backup
        Make backup of ``<calc_dir>`` to ``<calc_dir>.bak_<timestamp>_run_id_<run_id>``
        where ``<run_id>`` is the latest ``_run_id`` present in ``df``
    git
        Use ``git`` to commit all files written and changed by the current run
        (``_run_id``). Make sure to create a ``.gitignore`` manually before if
        needed.
    skip_dups
        Skip psets whose hash is already present in `df`. Useful when repeating
        (parts of) a study.
    capture_logs
        {'db', 'file', 'db+file', None}
        Redirect stdout and stderr generated in ``func()`` to database ('db')
        column ``_logs``, file ``<calc_dir>/<pset_id>/logs.txt``, or both. If
        ``None`` then do nothing (default). Useful for capturing per-pset log
        text, e.g. ``print()`` calls in `func` will be captured.
    fill_value
        NA value used for missing values in the database DataFrame.

    Returns
    -------
    df
        The database build from `params`.
    """

    # Don't in-place alter dicts in params we get as input.
    params = copy.deepcopy(params)

    # Wash params thru type machinery to ensure that types are the same as the
    # ones we will have in the DataFrame later on. Else _pset_hash can be
    # wrong (e.g. if columns contain NA-type values which are not
    # `fill_value`).
    params = df_extract_dicts(
        df_ensure_dtypes(
            pd.DataFrame(params, dtype=object), fill_value=fill_value
        ),
        py_types=False,
    )

    git_enter(git)

    if simulate:
        calc_dir = _setup_simulate_dir(
            calc_dir=calc_dir,
            database_basename=database_basename,
        )
    database_fn = pj(calc_dir, database_basename)

    if df is None:
        if os.path.exists(database_fn):
            df = df_read(database_fn).astype(object)
        else:
            df = pd.DataFrame(dtype=object)

    if len(df) == 0:
        pset_seq_old = -1
        run_seq_old = -1
    else:
        # int(): convert np.int64 back to Python int
        pset_seq_old = int(df._pset_seq.values.max())
        run_seq_old = int(df._run_seq.values.max())

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

    for pset in params:
        pset["_pset_hash"] = pset_hash(pset)

    if len(df) > 0:
        df = df_update_pset_cols(
            df,
            pset_cols=set(itertools.chain.from_iterable(params)),
            fill_value=fill_value,
        )
        if skip_dups:
            params = filter_params_dup_hash(params, df._pset_hash.values)

    run_id = get_uuid(existing=df._run_id.values if len(df) > 0 else [])
    pset_ids = get_many_uuids(
        len(params), existing=df._pset_id.values if len(df) > 0 else []
    )
    for ii, (pset, pset_id) in enumerate(zip(params, pset_ids)):
        pset["_pset_id"] = pset_id
        pset["_run_seq"] = run_seq_old + 1
        pset["_pset_seq"] = pset_seq_old + ii + 1
        pset["_run_id"] = run_id
        pset["_calc_dir"] = calc_dir

    if capture_logs is not None:
        func = partial(
            capture_logs_wrapper, func=func, capture_logs=capture_logs
        )

    func = partial(
        func_wrapper,
        func=func,
        tmpsave=tmpsave,
        verbose=verbose,
        simulate=simulate,
    )

    if (poolsize is None) and (dask_client is None):
        results = list(map(func, params))
    else:
        assert [poolsize, dask_client].count(None) == 1, (
            "Use either poolsize or dask_client."
        )
        if dask_client is None:
            with mp.Pool(poolsize) as pool:
                results = pool.map(func, params)
        else:
            futures = dask_client.map(func, params)
            results = dask_client.gather(futures)

    df = df_ensure_dtypes(
        pd.concat(
            (df, pd.DataFrame(results, dtype=object)),
            sort=False,
            ignore_index=True,
        ),
        fill_value=fill_value,
    )

    if save:
        df_write(database_fn, df)

    git_exit(git, df)

    return df


# -----------------------------------------------------------------------------
# (HPC cluster) batch runs using file templates
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

    def fill(self, pset, mode="jinja"):
        try:
            txt = file_read(self.filename)
            if mode == "dollar":
                return string.Template(txt).substitute(pset)
            elif mode == "jinja":
                return (
                    jinja2.Environment(undefined=jinja2.StrictUndefined)
                    .from_string(txt)
                    .render(pset)
                )
            else:
                raise ValueError(f"template {mode}= not supported")
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


def prep_batch(
    params: Sequence[dict],
    *,
    calc_templ_dir: str = "templates/calc",
    machine_templ_dir: str = "templates/machines",
    git: bool = False,
    write_pset: bool = False,
    template_mode: str = "jinja",
    **kwds,
) -> pd.DataFrame:
    """
    Write files based on templates.

    Parameters
    ----------
    params
        See :func:`run`
    calc_templ_dir, machine_templ_dir
        Dir with templates.
    git
        Use git to commit local changes.
    write_pset
        Write the input `pset` to ``<calc_dir>/<pset_id>/pset.pk``.
    template_mode
        'dollar' or 'jinja'
    **kwds
        Passed to :func:`run`.

    Returns
    -------
    df
        The database build from `params`.
    """
    git_enter(git)

    # Same defaults as in run()
    calc_dir = kwds.pop("calc_dir", CALC_DIR)
    database_basename = kwds.pop("database_basename", DATABASE_BASENAME)

    # Catch simulate flag, don't pass that thru to run() b/c there this
    # prevents func() from being executed, but that's what we always want here
    # since it writes only input files. Instead, just set calc_dir =
    # calc_dir_sim, copy the database as in run() and go. Don't copy the
    # run_*.sh scripts b/c they are generated afresh anyway.
    if kwds.pop("simulate", False):
        calc_dir = _setup_simulate_dir(
            calc_dir=calc_dir,
            database_basename=database_basename,
        )

    calc_templates = gather_calc_templates(calc_templ_dir)
    machines = gather_machines(machine_templ_dir)
    templates = calc_templates + [m.template for m in machines]

    def func(pset):
        for template in templates:
            file_write(
                pj(calc_dir, pset["_pset_id"], template.targetname),
                template.fill(pset, mode=template_mode),
            )
            if write_pset:
                pickle_write(pj(calc_dir, pset["_pset_id"], "pset.pk"), pset)
        return {}

    df = run(
        func,
        params,
        git=False,
        simulate=False,
        calc_dir=calc_dir,
        database_basename=database_basename,
        **kwds,
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
