import io
import os
import re
import shutil
import string
import sys
import tempfile
import pickle
import subprocess
from itertools import product
from contextlib import nullcontext
import importlib
import uuid
import copy
from functools import partial
import textwrap

import pandas as pd
import numpy as np
import pytest
import joblib
from packaging.version import parse as parse_version

import psweep as ps

pj = os.path.join
here = os.path.abspath(os.path.dirname(__file__))


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def system(cmd):
    return ps.system(cmd).stdout.decode()


def func_a(pset):
    # We need to multiply by a float here to make sure the 'result_' column has
    # float dtype. Else the column will be cast to float once we add NaNs,
    # which breaks df.equals(other_df) .
    return {"result_": pset["a"] * 10.0}


def find_examples(skip=["batch_dask"]):
    found_paths = []
    dr = os.path.abspath(f"{here}/../examples")
    for basename in os.listdir(dr):
        do_skip = False
        path = pj(dr, basename)
        for pattern in skip:
            if pattern in basename:
                print(f"skipping {path} because of skip {pattern=}")
                do_skip = True
                break
        if not do_skip:
            found_paths.append(path)
    return found_paths


class DummyClass:
    """Dummy custom type used in test_dotdict(). As always, must be defined
    outside b/c Python can't pickle stuff defined in the same scope.
    """

    x = 55

    def __eq__(self, other):
        return self.x == other.x

    def __hash__(self):
        return self.x


def dummy_func(a):
    b = a * 3
    return b


# ----------------------------------------------------------------------------
# test function
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("path", find_examples())
def test_run_all_examples(path):
    # Running examples via shell will swallow all warnings, but hey, one
    # problem at a time.
    print(f"running example: {path}")
    if path.endswith(".py"):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"""
                cp {path} {tmpdir}/ && cd {tmpdir} && \
                python3 {path}
            """
            print(system(cmd))
    elif os.path.isdir(path):
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copytree(path, tmpdir, dirs_exist_ok=True)
            cmd = f"cd {tmpdir} &&./run_example.sh &&./clean.sh"
            print(system(cmd))


def test_shell_call_fail():
    """
    When calling multiple shell commands, we need to chain them using
      cmd1 && cmd2 && ... && cmdN
    rather than
      cmd1; cmd2; ...; cmdN
    In the latter case, only cmdN's exit code determines the returncode, thus
    hiding any previously failed commands.
    """
    # this must pass
    system("ls; thiswillfail but be hidden; ls; pwd")

    # all others must fail
    with pytest.raises(subprocess.CalledProcessError):
        system("ls && thiswillfail and break the chain && ls && pwd")

    # several types of script content that must fail
    txts = []
    txts.append(
        """
echo "exit 1"
exit 1
"""
    )

    txts.append(
        """
set -eux
exit 1111
"""
    )

    txts.append(
        """
set -eux
randomfoocommanddoesntexist
"""
    )

    txts.append(
        """
set -eux
python3 -c "raise Exception('foo')"
"""
    )

    basename = "test.sh"
    for txt in txts:
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(subprocess.CalledProcessError):
                ps.file_write(pj(tmpdir, basename), txt)
                print(system(f"cd {tmpdir}; sh {basename}"))


def test_shell_call():
    print(system("ls"))
    with tempfile.TemporaryDirectory() as tmpdir:
        txt = "ls"
        basename = "test.sh"
        ps.file_write(pj(tmpdir, basename), txt)
        print(system(f"cd {tmpdir}; sh {basename}"))


def test_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        params = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
        calc_dir = f"{tmpdir}/calc"

        # run two times, updating the database, the second time,
        # also write tmp results
        df = ps.run(func_a, params, calc_dir=calc_dir)
        assert len(df) == 4
        assert len(df._run_id.unique()) == 1
        assert len(df._pset_id) == 4
        assert len(df._pset_id.unique()) == 4
        assert len(df._pset_hash) == 4
        assert len(df._pset_hash.unique()) == 4
        # For serial runs, this is guaranteed. For parallel runs, probably
        # also, as long as we use only concurrent.futures style APIs, avoid
        # map_async() and get results back in order. Should be safe with
        # multiprocessing, not sure about dask, though.
        assert (df._pset_seq.values == df.index.values).all()

        df = ps.run(
            func_a, params, calc_dir=calc_dir, poolsize=2, tmpsave=True
        )
        assert len(df) == 8
        assert len(df._run_id.unique()) == 2
        assert len(df._pset_id) == 8
        assert len(df._pset_id.unique()) == 8
        assert len(df._pset_hash) == 8
        assert len(df._pset_hash.unique()) == 4
        assert (df._pset_seq.values == df.index.values).all()
        assert set(df.columns) == set(
            [
                "_calc_dir",
                "_pset_id",
                "_run_id",
                "_pset_seq",
                "_run_seq",
                "_pset_hash",
                "_time_utc",
                "_pset_runtime",
                "_exec_host",
                "a",
                "result_",
            ]
        )

        dbfn = f"{calc_dir}/database.pk"
        assert os.path.exists(dbfn)
        assert df.equals(ps.df_read(dbfn))

        # tmp results of second run
        run_id = df._run_id.unique()[-1]
        df_sel = df[df._run_id == run_id]
        for pset_id in df_sel._pset_id.values:
            tmpsave_fn = f"{calc_dir}/tmpsave/{run_id}/{pset_id}.pk"
            assert os.path.exists(tmpsave_fn)
            read_dct = ps.pickle_read(tmpsave_fn)
            assert isinstance(read_dct, dict)
            assert (
                df_sel[df_sel._pset_id == pset_id].iloc[0].to_dict()
                == read_dct
            )


@pytest.mark.parametrize("use_disk", [True, False])
def test_run_skip_dups(use_disk):
    params = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
    Ctx = tempfile.TemporaryDirectory if use_disk else nullcontext
    with Ctx() as tmpdir:
        # calc_dir is passed but not used if save=use_disk=False, so it
        # doesn't matter that in this case tmpdir=None and so
        # calc_dir="None/calc" as long as it is a string. The dir doesn't have
        # to exist.
        calc_dir = f"{tmpdir}/calc"

        df1 = ps.run(
            func_a,
            params,
            calc_dir=calc_dir,
            save=use_disk,
        )
        assert len(df1) == 4
        assert len(df1._run_id.unique()) == 1
        assert len(df1._pset_id.unique()) == 4

        # Run again w/ same params, but now skip_dups=True. This
        # will cause no psets to be run. That's why df2 is equal to df1.
        df2 = ps.run(
            func_a,
            params,
            calc_dir=calc_dir,
            df=None if use_disk else df1,
            save=use_disk,
            skip_dups=True,
        )
        assert df2.equals(df1)

        # Now use params where a subset is new (last 2 entries).
        params = [{"a": 1}, {"a": 2}, {"a": 88}, {"a": 99}]
        df3 = ps.run(
            func_a,
            params,
            calc_dir=calc_dir,
            df=None if use_disk else df2,
            save=use_disk,
            skip_dups=True,
        )
        assert len(df3) == 6
        assert len(df3._run_id.unique()) == 2
        assert len(df3._pset_id.unique()) == 6
        assert (
            df3._pset_hash.to_list()
            == df1._pset_hash.to_list() + df3._pset_hash.to_list()[-2:]
        )


def test_simulate():
    with tempfile.TemporaryDirectory() as tmpdir:
        params = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
        params_sim = [{"a": 88}, {"a": 99}]
        calc_dir = f"{tmpdir}/calc"
        calc_dir_sim = calc_dir + ".simulate"

        df = ps.run(func_a, params, calc_dir=calc_dir)
        df_sim = ps.run(func_a, params_sim, calc_dir=calc_dir, simulate=True)
        dbfn = f"{calc_dir}/database.pk"
        dbfn_sim = f"{calc_dir_sim}/database.pk"

        assert len(df_sim) == 6
        assert len(df) == 4
        assert os.path.exists(dbfn)
        assert os.path.exists(dbfn_sim)
        assert df.equals(ps.df_read(dbfn))
        assert df_sim.equals(ps.df_read(dbfn_sim))

        assert df.iloc[:4].equals(df_sim.iloc[:4])
        assert np.isnan(df_sim.result_.values[-2:]).all()

        df2 = ps.run(func_a, params_sim, calc_dir=calc_dir)
        assert len(df2) == 6
        assert df.iloc[:4].equals(df2.iloc[:4])
        assert (df2.result_.values[-2:] == np.array([880.0, 990.0])).all()


def test_is_seq():
    no = [{"a": 1}, io.IOBase(), "123"]
    yes = [[1, 2], {1, 2}, (1, 2)]
    for obj in no:
        print(obj)
        assert not ps.is_seq(obj)
    for obj in yes:
        print(obj)
        assert ps.is_seq(obj)


def test_df_io():
    from pandas.testing import assert_frame_equal

    letters = string.ascii_letters
    ri = np.random.randint
    rn = np.random.rand

    # random string
    def rs(n):
        return "".join(letters[ii] for ii in ri(0, len(letters), n))

    for fmt in ["pickle", "json"]:
        df = pd.DataFrame()
        for _ in range(2):
            vals = [
                ri(0, 100),
                rs(5),
                np.nan,
                f'"{rs(5)}"',
                f"'{rs(5)}'",
                (ri(0, 99), rn(), f"{rs(5)}"),
                [ri(0, 99), rn(), f"{rs(5)}"],
                rn(),
                rn(5),
                rn(5, 5),
                list(rn(5)),
                {"a": 1, "b": 3, "c": [1, 2, 3]},
            ]
            if fmt == "pickle":
                vals += [
                    True,
                    False,
                    None,
                    set(ri(0, 99, 10)),
                ]
            row = pd.DataFrame([dict(zip(letters, vals))])
            df = pd.concat((df, row), ignore_index=True)

        if fmt == "json":
            for orient in [
                None,
                "split",
                "records",
                "index",
                "columns",
                "_default_",
            ]:
                print("orient: ", orient)
                with tempfile.NamedTemporaryFile() as fd:
                    if orient != "_default_":
                        ps.df_write(fd.name, df, fmt=fmt, orient=orient)
                        read = ps.df_read(fd.name, fmt=fmt, orient=orient)
                    else:
                        ps.df_write(fd.name, df, fmt=fmt)
                        read = ps.df_read(fd.name, fmt=fmt)
                    assert_frame_equal(df, read, check_exact=False)
        elif fmt == "pickle":
            with tempfile.NamedTemporaryFile() as fd:
                ps.df_write(fd.name, df, fmt=fmt)
                read = ps.df_read(fd.name, fmt=fmt)
                assert_frame_equal(df, read)
        else:
            raise Exception("unknown fmt")


def test_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        params = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
        calc_dir = f"{tmpdir}/calc"
        dbfn = f"{calc_dir}/database.pk"

        ps.run(func_a, params, calc_dir=calc_dir, save=False)
        assert not os.path.exists(dbfn)
        assert os.listdir(tmpdir) == []

        ps.run(func_a, params, calc_dir=calc_dir, save=True)
        assert os.path.exists(dbfn)
        assert os.listdir(tmpdir) != []


def test_merge_dicts():
    a = {"a": 1}
    b = {"b": 2}
    c = {"c": 3}

    # API
    assert a == ps.merge_dicts(a)
    m1 = ps.merge_dicts(a, b, c)
    m2 = ps.merge_dicts([a, b, c])

    # correct merge
    assert set(m1.keys()) == set(m2.keys())
    assert set(m1.values()) == set(m2.values())
    assert set(m1.keys()) == set(("a", "b", "c"))
    assert set(m1.values()) == set((1, 2, 3))

    # left-to-right
    a = {"a": 1}
    b = {"a": 2}
    c = {"a": 3}
    m = ps.merge_dicts(a, b)
    assert m == {"a": 2}
    m = ps.merge_dicts(a, b, c)
    assert m == {"a": 3}


def test_scripts():
    with tempfile.TemporaryDirectory() as tmpdir:
        params = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
        calc_dir = f"{tmpdir}/calc"
        ps.run(func_a, params, calc_dir=calc_dir)

        db = pj(calc_dir, "database.pk")
        print(system(f"psweep-db2json -o columns {db}"))
        print(system(f"psweep-db2table -i -a -f simple {db}"))


def test_backup():
    def func(pset):
        # write stuff to calc_dir
        dr = pj(pset["_calc_dir"], pset["_pset_id"])
        ps.makedirs(dr)
        fn = pj(dr, "foo")
        with open(fn, "w") as fd:
            fd.write(pset["_pset_id"])
        return {"result_": pset["a"] * 10}

    with tempfile.TemporaryDirectory() as tmpdir:
        calc_dir = pj(tmpdir, "calc")
        params = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]

        # First run. backup does nothing yet
        df0 = ps.run(func, params, calc_dir=calc_dir)
        unq = df0._run_id.unique()
        assert len(unq) == 1
        run_id_0 = unq[0]

        # Second run. This time, test backup.
        df1 = ps.run(func, params, calc_dir=calc_dir, backup=True)
        rex = re.compile(r"calc.bak_[0-9-]+T[0-9:\.]+Z_run_id.+")
        found = False
        files = os.listdir(tmpdir)
        for name in os.listdir(tmpdir):
            if rex.search(name) is not None:
                backup_dir = pj(tmpdir, name)
                found = True
                break
        assert found, f"backup dir matching {rex} not found in:\n{files}"
        print(os.listdir(backup_dir))
        msk = df1._run_id == run_id_0
        assert len(df1[msk]) == len(df0)
        assert len(df1) == 2 * len(df0)
        assert len(df1._run_id.unique()) == 2
        for pset_id in df1[msk]._pset_id:
            tgt = pj(backup_dir, pset_id, "foo")
            assert os.path.exists(tgt)
            with open(tgt) as fd:
                fd.read().strip() == pset_id
        assert os.path.exists(pj(backup_dir, "database.pk"))


def test_pass_df_interactive():
    def df_cmp(dfa, dfb):
        assert (dfa.a.values == dfb.a.values).all()
        assert (dfa.result_.values == dfb.result_.values).all()
        assert (dfa._pset_seq.values == dfb._pset_seq.values).all()
        assert (dfa._run_seq.values == dfb._run_seq.values).all()
        assert (dfa._pset_hash.values == dfb._pset_hash.values).all()

    with tempfile.TemporaryDirectory() as tmpdir:
        calc_dir = pj(tmpdir, "calc")
        params = ps.plist("a", [1, 2, 3, 4])

        # no db disk write for now, test passing in no df, df=None and empty df
        df1_1 = ps.run(func_a, params, calc_dir=calc_dir, save=False)
        df1_2 = ps.run(func_a, params, calc_dir=calc_dir, save=False, df=None)
        df_cmp(df1_1, df1_2)
        df1_3 = ps.run(
            func_a, params, calc_dir=calc_dir, save=False, df=pd.DataFrame()
        )
        df_cmp(df1_1, df1_3)

        # still no disk write, pass in df1 and extend
        df1 = df1_3
        df2 = ps.run(func_a, params, calc_dir=calc_dir, save=False, df=df1)
        assert not os.path.exists(pj(tmpdir, "calc"))
        assert len(df2) == 2 * len(df1)
        assert (df2.a.values == np.tile(df1.a.values, 2)).all()
        assert (
            df2._pset_hash.values == np.tile(df1._pset_hash.values, 2)
        ).all()

        # df2 again, but now write to disk and read (ignore that run()
        # also returns it)
        ps.run(func_a, params, calc_dir=calc_dir, df=df1)
        df2_disk = ps.df_read(pj(calc_dir, "database.pk"))
        df_cmp(df2, df2_disk)


def test_df_filter_conds():
    df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10) + 4})

    # logical and
    conds = [df.a > 3, df.b < 9, df.a % 2 == 0]
    df_ref = df[conds[0] & conds[1] & conds[2]]
    assert df_ref.equals(ps.df_filter_conds(df, conds))
    for filt in [lambda x: x, lambda x: x.to_numpy(), lambda x: x.to_list()]:
        # use iterator map(...)
        assert df_ref.equals(ps.df_filter_conds(df, map(filt, conds)))

    # logical or. For the conds above, OR-ing them happens to result in the
    # original df w/o and reduction.
    assert df.equals(ps.df_filter_conds(df, conds, op="or"))


# Uncomment to run test_dotdict using one of those implementations.

# https://github.com/cdgriffith/Box
# pip install python-box
##from box import Box
##ps.dotdict = Box

# https://github.com/drgrib/dotmap
# pip install dotmap
##from dotmap import DotMap
##ps.dotdict = DotMap


@pytest.mark.skipif(
    not hasattr(ps, "dotdict"), reason="psweep.dotdict not defined"
)
def test_dotdict():
    """Test a possible implementation for a dict with optional dot attr access.
    See e.g. [1]  for all the stunts people have pulled so far (including
    ourselves [2]!!). The idea is to eventually use that for

    * psets instead of plain dicts
    * general purpose dot access container type (e.g. for param study driver
      scripts)

    None of the present hacks meets all our needs

    * optional dot access
    * update __repr__ after attr setting (e.g. d=dotdict(...); d.x=4)
    * can be pickled

    except for those that live in an external package (box, dotmap), but
    getting some random pypi package as a dep is not worth it. Maybe we'll
    stumble upon a small and neat implementation that we can add to psweep.
    Until then, just stick to plain dicts.


    [1] https://stackoverflow.com/a/23689767
    [2] https://github.com/elcorto/pwtools/blob/master/pwtools/batch.py#L735
    """
    # psweep.py::
    #
    #     class dotdict(dict):
    #         """Not good enough, can't be pickled. So sad!"""
    #         __getattr__ = dict.get
    #         __setattr__ = dict.__setitem__
    #         __delattr__ = dict.__delitem__
    #         __dir__ = dict.keys

    f = DummyClass()
    ref = dict(a=1, b=2, c=np.sin, d=f, e=DummyClass)

    def dct_cmp(ref, ddict):
        assert ref == ddict
        assert set(ref.keys()) == set(ddict.keys())
        assert set(ref.values()) == set(ddict.values())
        if isinstance(ddict, ps.dotdict):
            ddict.a
            ddict.b
            ddict.c
            ddict.d
            ddict.e

    for ddict in [ps.dotdict(ref), ps.dotdict(**ref)]:
        dct_cmp(ref, ddict)
        # Test whether dotdict implementation can be pickled.
        dct_cmp(ref, pickle.loads(pickle.dumps(ddict)))


def test_pset_hash():
    dref = dict(a=1, b=dict(c=2, d=[1, 2, "a"]))
    d1 = dict(b=dict(c=2, d=[1, 2, "a"]), a=1)
    d2 = dict(a=1, b=dict(d=[1, 2, "a"], c=2))

    assert ps.pset_hash(dref) == "c41c669212f58293fb88aba643b24a995d9aff1a"
    assert ps.pset_hash(d1) == ps.pset_hash(dref)
    assert ps.pset_hash(d2) == ps.pset_hash(dref)

    h_ref_np_v1 = "32522716b0514eb8b2677a917888c4ca21f792da"
    h_ref_np_v2 = "6f5ceea02d337dac4914401d5cb53476eb68493c"
    h_val = ps.pset_hash(dict(a=np.sin))
    if parse_version(np.__version__).major == 1:
        assert h_val == h_ref_np_v1
    else:
        assert h_val == h_ref_np_v2

    # These are the correct hashes (tested: ipython or run this function in a
    # script). But pytest's deep introspection "magic" must do something to the
    # DummyClass class such that its hash changes, thank you very much. So we just
    # test that hashing them works.
    #
    ##assert (
    ##    ps.pset_hash(dict(a=DummyClass))
    ##    == "f7a936832c167d5dc1c9574b937822ad0853c00d"
    ##)
    ##assert (
    ##    ps.pset_hash(dict(a=DummyClass()))
    ##    == "7c78a7c995e1cf7a2f1f68f29090b173c9b930fa"
    ##)
    ps.pset_hash(dict(a=DummyClass))
    ps.pset_hash(dict(a=DummyClass()))


def test_pset_hash_skip_cols():
    d_no_pre_post = dict(a=1, b=2)
    d_pre = dict(a=1, b=2, _c=3)
    d_post = dict(a=1, b=2, d_=4)
    d_pre_post = dict(a=1, b=2, _c=3, d_=4)

    f_hash = lambda x: joblib.hash(x, hash_name=ps.PSET_HASH_ALG)

    assert ps.pset_hash(d_pre_post) == f_hash(d_no_pre_post)

    assert ps.pset_hash(d_no_pre_post) == ps.pset_hash(d_pre)
    assert ps.pset_hash(d_no_pre_post) == ps.pset_hash(d_post)
    assert ps.pset_hash(d_no_pre_post) == ps.pset_hash(d_pre_post)

    assert ps.pset_hash(d_pre_post, skip_prefix_cols=False) == f_hash(d_pre)
    assert ps.pset_hash(d_pre_post, skip_postfix_cols=False) == f_hash(d_post)
    assert ps.pset_hash(
        d_pre_post, skip_prefix_cols=False, skip_postfix_cols=False
    ) == f_hash(d_pre_post)

    with pytest.deprecated_call():
        ps.pset_hash(d_pre_post, skip_special_cols=True)


@pytest.mark.parametrize(
    "non_py_type",
    [
        2,
        pytest.param(np.int64(2), marks=pytest.mark.xfail),
        pytest.param(np.uint64(2), marks=pytest.mark.xfail),
        pytest.param(np.float64(2.0), marks=pytest.mark.xfail),
    ],
)
def test_pset_hash_recalc_from_df(non_py_type):
    params = ps.plist(
        "a", [1, "3", np.sin, [], None, 1.23, dummy_func, non_py_type]
    )
    df = ps.run(lambda pset: {}, params, save=False)
    for idx, row in df.iterrows():
        df.at[idx, "_pset_hash_new"] = ps.pset_hash(row.to_dict())

    assert (df._pset_hash.values == df._pset_hash_new.values).all()


def test_param_build():
    a = ps.plist("a", [1, 2])
    b = ps.plist("b", [77, 88, 99])

    assert a == ps.pgrid([a])
    with pytest.raises(AssertionError):
        ps.pgrid(a)

    abref = [
        {"a": 1, "b": 77},
        {"a": 1, "b": 88},
        {"a": 1, "b": 99},
        {"a": 2, "b": 77},
        {"a": 2, "b": 88},
        {"a": 2, "b": 99},
    ]

    assert abref == ps.pgrid(a, b)
    assert abref == ps.pgrid([a, b])
    assert abref == ps.itr2params(product(a, b))

    # a and c together, no product() here so it is an error to use pgrid(a,c)
    # which calls product(): pgrid(a,c) = itr2params(product(a,c)). We have
    # only
    #
    # >>> list(zip(a,c))
    # [({'a': 1}, {'c': 11}),
    #  ({'a': 2}, {'c': 22})]
    # >>> ps.itr2params(zip(a,c))
    # [{'a': 1, 'c': 11},
    #  {'a': 2, 'c': 22}]
    c = ps.plist("c", [11, 22])
    acref = [{"a": 1, "c": 11}, {"a": 2, "c": 22}]
    assert acref == ps.itr2params(zip(a, c))
    assert acref == ps.pgrid([zip(a, c)])
    with pytest.raises(AssertionError):
        ps.pgrid(zip(a, c))

    d = ps.plist("d", [66, 77, 88, 99])
    acdref = ps.itr2params(product(zip(a, c), d))
    assert acdref == ps.pgrid([zip(a, c), d])
    assert acdref == ps.pgrid(zip(a, c), d)


def test_itr():
    @ps.itr
    def func(args):
        return [a for a in args]

    def gen(inp):
        for x in inp:
            yield x

    assert [1] == func(1)
    assert [1] == func([1])
    assert [[1]] == func([[1]])
    assert [1, 2] == func(1, 2)
    assert [1, 2] == func([1, 2])
    assert [[1, 2]] == func([[1, 2]])
    assert [1, 2] == func(gen([1, 2]))


def test_filter_params_unique():
    params = [
        dict(a=1, b=2, c=3),  # same!
        dict(a=1, b=2, c=3),  # same!
        dict(a=1, b=2, c=444),
    ]

    params_filt = ps.filter_params_unique(params)
    assert len(params_filt) == 2
    for idx in [0, 2]:
        assert params[idx] in params_filt


def test_filter_params_dup_hash():
    params = [
        dict(a=1, b=2, c=3),  # same!
        dict(a=1, b=2, c=3),  # same!
        dict(a=1, b=2, c=333),
        dict(a=1, b=2, c=444),
    ]

    hashes = [ps.pset_hash(pset) for pset in params]
    params_filt = ps.filter_params_dup_hash(params, hashes)
    assert len(params_filt) == 0

    hashes = [ps.pset_hash(pset) for pset in params[:3]]
    params_filt = ps.filter_params_dup_hash(params, hashes)
    assert len(params_filt) == 1
    assert params_filt[0] == params[-1]


def test_stargrid():
    a = ps.plist("a", [1, 2, 3])
    b = ps.plist("b", [77, 88, 99])
    c = ps.plist("c", [np.sin, np.cos, DummyClass, DummyClass()])
    const = dict(a=1, b=77)

    # API
    ps.stargrid(const, [a, b])
    ps.stargrid(const=const, vary=[a, b])

    params = ps.stargrid(const, [a, b], skip_dups=False)
    assert len(params) == 6
    params = ps.stargrid(const, [a, b])
    assert len(params) == 5
    params = ps.stargrid(const, [c])
    assert len(params) == len(c)
    params = ps.stargrid(dict(), [a])
    assert params == a

    vary_labels = ["aa", "bb"]
    params = ps.stargrid(const, [a, b], vary_labels=vary_labels)
    for pset in params:
        assert "_vary" in pset.keys()
        assert pset["_vary"] in vary_labels

    params = ps.stargrid(
        const, [a, b], vary_labels=vary_labels, vary_label_col="study"
    )
    for pset in params:
        assert "study" in pset.keys()
        assert "_vary" not in pset.keys()
        assert pset["study"] in vary_labels


def test_intspace():
    assert (ps.intspace(0, 4, 5) == np.array(range(5))).all()
    # array([0.  , 1.25, 2.5 , 3.75, 5.  ]) -> round to int
    assert (ps.intspace(0, 5, 5) == np.array([0, 1, 2, 4, 5])).all()
    # array([0, 1, 2, 3, 4, 5])
    assert (ps.intspace(0, 5, 20) == np.array(range(6))).all()


def test_logspace_defaults():
    start = np.random.rand()
    stop = 5 * start
    log_func = np.log10
    ref = np.logspace(log_func(start), log_func(stop))
    np.testing.assert_allclose(ref, ps.logspace(start, stop))


@pytest.mark.parametrize("log_func, base", [(np.log10, 10), (np.log, np.e)])
def test_logspace_base_log_func(log_func, base):
    start = np.random.rand()
    stop = 5 * start
    ref = np.logspace(log_func(start), log_func(stop), base=base, num=20)
    # offset=0
    np.testing.assert_allclose(
        ref, ps.logspace(start, stop, num=20, base=base, log_func=log_func)
    )


@pytest.mark.parametrize("offset", [1, 2.345])
def test_logspace_offset(offset):
    start = np.random.rand()
    stop = 5 * start
    log_func = np.log10
    ref = np.logspace(log_func(start), log_func(stop))
    val = ps.logspace(start, stop, offset=offset)
    np.testing.assert_allclose(val[0], start)
    np.testing.assert_allclose(val[-1], stop)
    assert not np.allclose(ref, val)


def test_prep_batch():
    a = ps.plist("param_a", [1, 2, 3])
    b = ps.plist("param_b", ["xx", "yy"])
    params = ps.pgrid(a, b)
    template_dir = f"{here}/../examples/batch_templates_git/templates"
    with tempfile.TemporaryDirectory() as tmpdir:
        # run 0
        df = ps.prep_batch(
            params,
            calc_dir=f"{tmpdir}/calc",
            calc_templ_dir=f"{template_dir}/calc",
            machine_templ_dir=f"{template_dir}/machines",
            database_dir=f"{tmpdir}/db",
            write_pset=True,
        )
        print(system(f"tree {tmpdir}"))

        assert os.path.exists(f"{tmpdir}/db/database.pk")
        assert os.path.exists(f"{tmpdir}/calc/run_local.sh")
        assert os.path.exists(f"{tmpdir}/calc/run_cluster.sh")

        n_param = len(params)
        for txt in [
            ps.file_read(f"{tmpdir}/calc/run_cluster.sh"),
            ps.file_read(f"{tmpdir}/calc/run_local.sh"),
        ]:
            assert "# run_seq=0 pset_seq=0" in txt
            assert f"# run_seq=0 pset_seq={n_param - 1}" in txt

        for pset_id in df._pset_id.values:
            for name in [
                "run.py",
                "pset.pk",
                "jobscript_local",
                "jobscript_cluster",
            ]:
                assert os.path.exists(f"{tmpdir}/calc/{pset_id}/{name}")

        # run 1
        df = ps.prep_batch(
            params,
            calc_dir=f"{tmpdir}/calc",
            calc_templ_dir=f"{template_dir}/calc",
            machine_templ_dir=f"{template_dir}/machines",
            database_dir=f"{tmpdir}/db",
            write_pset=True,
        )

        for txt in [
            ps.file_read(f"{tmpdir}/calc/run_cluster.sh"),
            ps.file_read(f"{tmpdir}/calc/run_local.sh"),
        ]:
            assert "# run_seq=0 pset_seq=0" in txt
            assert f"# run_seq=0 pset_seq={n_param - 1}" in txt
            assert f"# run_seq=1 pset_seq={n_param}" in txt
            assert f"# run_seq=1 pset_seq={2*n_param - 1}" in txt


# We can't use
#
#   @pytest.mark.skipif(
#       importlib.util.find_spec("dask.distributed") is None,
#       reason="dask.distributed not found",
#   )
#
# If both dask and distributed (import as dask.distributed) are *not*
# installed, find_spec("dask.distributed") errors out saying that dask is not
# installed. This is because (from the docs):
#
#   importlib.util.find_spec(name, package=None)
#
#   If name is for a submodule (contains a dot), the parent module is
#   automatically imported.
#
# OK, so instead of testing if dask is installed, it just tries to import it.
#
# We can't solve this by using a second skipif to test for dask first.
#
#   @pytest.mark.skipif(
#       importlib.util.find_spec("dask") is None,
#       reason="dask not found",
#   )
#   @pytest.mark.skipif(
#       importlib.util.find_spec("dask.distributed") is None,
#       reason="dask.distributed not found",
#   )
#
# This doesn't work because pytest seems to always test both skipifs, no matter
# which wrapper comes first, so the order is irrelevant.
#
# The only solution we have so far is the one below. Since "foo or bar" is
# evaluated left-to-right, we first check if dask is installed and stop if not.
#
@pytest.mark.skipif(
    (importlib.util.find_spec("dask") is None)
    or (importlib.util.find_spec("dask.distributed") is None),
    reason="dask or dask.distributed not found",
)
def test_dask_local_cluster():
    from dask.distributed import Client, LocalCluster

    # LocalCluster is default if no cluster is provided
    client = Client(dashboard_address=None)

    params = ps.plist("a", [1, 2, 3])
    df = ps.run(func_a, params, dask_client=client, save=False)
    assert len(df) == 3
    client.close()

    # We don't need the dask dashboard in tests. Setting dashboard_address=None
    # should turn the dashboard off. But that doesn't work
    # (https://github.com/dask/distributed/issues/8136). We see warnings like
    #   UserWarning: Port 8787 is already in use. Perhaps you already have a
    #   cluster running? ...
    # as a result, if we run tests in parallel.
    cluster = LocalCluster(dashboard_address=None)
    client = Client(cluster)

    df = ps.run(func_a, params, dask_client=client, save=False, df=df)
    assert len(df) == 6


def test_pickle_io():
    obj = dict(a=1, b=DummyClass(), c=np.sin)
    hsh = lambda obj: joblib.hash(obj, hash_name="sha1")
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = f"{tmpdir}/path/that/has/to/be/created/file.pk"
        ps.pickle_write(fn, obj)
        assert hsh(obj) == hsh(ps.pickle_read(fn))


def test_file_io():
    txt = "some random text"
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = f"{tmpdir}/path/that/has/to/be/created/file.txt"
        ps.file_write(fn, txt)
        assert txt == ps.file_read(fn)


def test_single_uuid():
    existing = [str(uuid.uuid4()) for _ in range(100)]
    ret = ps.get_uuid()
    # 'b5233ba3-25eb-40ac-b2b1-d1babbacd904'
    assert isinstance(ret, str)
    assert len(ret) == 36
    assert re.match("([a-z0-9]+-){4}[a-z0-9]", ret) is not None
    ret = ps.get_uuid(existing=existing)
    assert ret not in existing


def test_many_uuids():
    existing = [str(uuid.uuid4()) for _ in range(100)]
    ret = set(ps.get_many_uuids(100))
    assert len(ret) == 100

    ret = set(ps.get_many_uuids(100, existing=existing))
    assert len(ret) == 100
    assert len(ret & set(existing)) == 0


def test_run_not_alter_params():
    params = ps.plist("a", [1, 2, 3])
    params_copy = copy.deepcopy(params)
    hsh = joblib.hash(params, hash_name="sha1")
    df = ps.run(func_a, params, save=False)
    df = ps.run(func_a, params, save=False, df=df)
    assert params == params_copy
    assert hsh == joblib.hash(params, hash_name="sha1")
    for pset in params:
        for key in pset.keys():
            assert not key.startswith("_")


class TestCaptureLogs:
    def get_dask_client(self, n):
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(dashboard_address=None)
        cluster.scale(n=n)
        client = Client(cluster)
        return client

    @staticmethod
    def get_print_funcs():
        fout = lambda msg: sys.stdout.write(msg)
        ferr = lambda msg: sys.stderr.write(msg)
        return fout, ferr

    @staticmethod
    def get_print_funcs_fail(sout=sys.stdout, serr=sys.stderr):
        """This case will fail, i.e. printed text will disappear into the
        nirvana. The reason is that the redirect machinery in run() will
        dynamically redirect sys.sydout and sys.stderr at runtime. However
        here, we have bound those streams to the sout and serr variables at
        parse time before run() can touch them, so they won't be affected.

        This is an example of user code where the redirect feature cannot be
        used.
        """
        fout = lambda msg: sout.write(msg)
        ferr = lambda msg: serr.write(msg)
        return fout, ferr

    def func_with_logs(self, pset, f_get_print_funcs=None):
        print(f"{pset=}")
        print("txt on stdout")
        print("txt on stderr", file=sys.stderr)
        sys.stdout.write("stdout direct\n")
        sys.stderr.write("stderr direct\n")
        fout, ferr = f_get_print_funcs()
        fout("txt from fout\n")
        ferr("txt from ferr")
        return dict()

    @staticmethod
    def assert_txt_content(txt, pset):
        # fmt: off
        ref_txt = textwrap.dedent(f"""\
            {pset=}
            txt on stdout
            txt on stderr
            stdout direct
            stderr direct
            txt from fout
            txt from ferr"""
            )
        # fmt: on
        assert ref_txt == txt

    def assert_log_content(self, df, run_id, in_file=False, in_db=False):
        dfs = df[df._run_id == run_id]
        assert len(dfs) == 100
        # Added after func() is called, so these won't be part of the pset
        # print in func().
        skip_cols = ["_pset_runtime", "_logs"]
        cols = [c for c in df.columns if c not in skip_cols]
        for calc_dir, pset_id in zip(
            dfs._calc_dir.values, dfs._pset_id.values
        ):
            pset = dfs.loc[dfs._pset_id == pset_id].iloc[0][cols].to_dict()
            if in_db:
                db_txt = dfs.loc[dfs._pset_id == pset_id, "_logs"].iloc[0]
                assert len(db_txt) > 0
                self.assert_txt_content(db_txt, pset)
            fn = pj(calc_dir, pset_id, "logs.txt")
            if in_file:
                assert os.path.exists(fn)
                txt = ps.file_read(fn)
                assert len(txt) > 0
                self.assert_txt_content(txt, pset)
                if in_db:
                    assert db_txt == txt
            else:
                assert not os.path.exists(fn)

    @staticmethod
    def get_parametrize_mark(test_func):
        xfail = pytest.mark.xfail(raises=AssertionError, strict=True)
        wrapper = pytest.mark.parametrize(
            "should_pass", [True, pytest.param(False, marks=xfail)]
        )
        return wrapper(test_func)

    @get_parametrize_mark
    def test_serial(self, should_pass):
        self.impl(use_dask=False, use_pool=False, should_pass=should_pass)

    @get_parametrize_mark
    def test_pool(self, should_pass):
        self.impl(use_dask=False, use_pool=True, should_pass=should_pass)

    @pytest.mark.skipif(
        (importlib.util.find_spec("dask") is None)
        or (importlib.util.find_spec("dask.distributed") is None),
        reason="dask or dask.distributed not found",
    )
    @get_parametrize_mark
    def test_dask(self, should_pass):
        self.impl(use_dask=True, use_pool=False, should_pass=should_pass)

    def impl(self, use_dask, use_pool, should_pass):
        assert not (use_dask and use_pool)
        dask_client = self.get_dask_client(n=3) if use_dask else None
        poolsize = 3 if use_pool else None
        func = partial(
            self.func_with_logs,
            f_get_print_funcs=self.get_print_funcs
            if should_pass
            else self.get_print_funcs_fail,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            calc_dir = f"{tmpdir}/calc"
            params = ps.plist("a", range(100))
            df = ps.run(
                func,
                params,
                calc_dir=calc_dir,
                capture_logs="file",
                dask_client=dask_client,
                poolsize=poolsize,
            )
            assert not hasattr(df, "_logs")
            self.assert_log_content(
                df, run_id=df._run_id.values[-1], in_file=True
            )

            df = ps.run(
                func,
                params,
                calc_dir=calc_dir,
                capture_logs="db",
                dask_client=dask_client,
                poolsize=poolsize,
            )
            assert hasattr(df, "_logs")
            self.assert_log_content(
                df, run_id=df._run_id.values[-1], in_db=True
            )

            df = ps.run(
                func,
                params,
                calc_dir=calc_dir,
                capture_logs="db+file",
                dask_client=dask_client,
                poolsize=poolsize,
            )
            assert hasattr(df, "_logs")
            self.assert_log_content(
                df, run_id=df._run_id.values[-1], in_file=True, in_db=True
            )


# We don't go the extra mile and verify the text output, but you may check like
# so:
#
# $ pytest -k verbose -vs
# tests/test_all.py::test_verbose[True]
#                                a
# 2023-11-22 19:41:49.736106634  1
#                                a
# 2023-11-22 19:41:49.739609957  2
#                                a
# 2023-11-22 19:41:49.742296457  3
# PASSED
# tests/test_all.py::test_verbose[False] PASSED
# tests/test_all.py::test_verbose[verbose2]
#                                                            _pset_id  a
# 2023-11-22 19:41:49.752099991  38a6820f-7e41-40af-b403-3647f44e8b9d  1
#                                                            _pset_id  a
# 2023-11-22 19:41:49.755405426  9ade6afb-1fd3-4d29-9e7a-0bacdb950994  2
#                                                            _pset_id  a
# 2023-11-22 19:41:49.757858753  80193006-56d4-47b1-be95-fe06cee2eec1  3
# PASSED
@pytest.mark.parametrize("verbose", [True, False, ["a", "_pset_id"]])
def test_verbose(verbose):
    params = ps.plist("a", [1, 2, 3])
    ps.run(func_a, params, save=False, verbose=verbose)


def test_version():
    assert isinstance(ps.__version__, str)
