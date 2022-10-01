import io
import os
import re
import shutil
import string
import tempfile
import pickle
import subprocess
from itertools import product
from contextlib import nullcontext

import pandas as pd
import numpy as np
import pytest

import psweep as ps

pj = os.path.join
here = os.path.abspath(os.path.dirname(__file__))

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def system(cmd):
    return ps.system(cmd).stdout.decode()


def func(pset):
    # We need to multiply by a float here to make sure the 'result' column has
    # float dtype. Else the column will be cast to float once we add NaNs,
    # which breaks df.equals(other_df) .
    return {"result": pset["a"] * 10.0}


# ----------------------------------------------------------------------------
# test function
# ----------------------------------------------------------------------------


def test_run_all_examples():
    dr = os.path.abspath(f"{here}/../examples")
    for basename in os.listdir(dr):
        path = pj(dr, basename)
        print(f"running example: {path}")
        if basename.endswith(".py"):
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
        df = ps.run_local(func, params, calc_dir=calc_dir)
        assert len(df) == 4
        assert len(df._run_id.unique()) == 1
        assert len(df._pset_id) == 4
        assert len(df._pset_id.unique()) == 4
        assert len(df._pset_hash) == 4
        assert len(df._pset_hash.unique()) == 4
        df = ps.run_local(
            func, params, calc_dir=calc_dir, poolsize=2, tmpsave=True
        )
        assert len(df) == 8
        assert len(df._run_id.unique()) == 2
        assert len(df._pset_id) == 8
        assert len(df._pset_id.unique()) == 8
        assert len(df._pset_hash) == 8
        assert len(df._pset_hash.unique()) == 4
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
                "a",
                "result",
            ]
        )

        dbfn = f"{calc_dir}/database.pk"
        assert os.path.exists(dbfn)
        assert df.equals(ps.df_read(dbfn))

        # tmp results of second run
        run_id = df._run_id.unique()[-1]
        for pset_id in df[df._run_id == run_id]._pset_id:
            tmpsave_fn = f"{calc_dir}/tmpsave/{run_id}/{pset_id}.pk"
            assert os.path.exists(tmpsave_fn)


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

        df1 = ps.run_local(
            func,
            params,
            calc_dir=calc_dir,
            save=use_disk,
        )
        assert len(df1) == 4
        assert len(df1._run_id.unique()) == 1
        assert len(df1._pset_id.unique()) == 4

        # Run again w/ same params, but now skip_dups=True. This
        # will cause no psets to be run. That's why df2 is equal to df1.
        df2 = ps.run_local(
            func,
            params,
            calc_dir=calc_dir,
            df=None if use_disk else df1,
            save=use_disk,
            skip_dups=True,
        )
        assert df2.equals(df1)

        # Now use params where a subset is new (last 2 entries).
        params = [{"a": 1}, {"a": 2}, {"a": 88}, {"a": 99}]
        df3 = ps.run_local(
            func,
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

        df = ps.run_local(func, params, calc_dir=calc_dir)
        df_sim = ps.run_local(
            func, params_sim, calc_dir=calc_dir, simulate=True
        )
        dbfn = f"{calc_dir}/database.pk"
        dbfn_sim = f"{calc_dir_sim}/database.pk"

        assert len(df_sim) == 6
        assert len(df) == 4
        assert os.path.exists(dbfn)
        assert os.path.exists(dbfn_sim)
        assert df.equals(ps.df_read(dbfn))
        assert df_sim.equals(ps.df_read(dbfn_sim))

        assert df.iloc[:4].equals(df_sim.iloc[:4])
        assert np.isnan(df_sim.result.values[-2:]).all()

        df2 = ps.run_local(func, params_sim, calc_dir=calc_dir)
        assert len(df2) == 6
        assert df.iloc[:4].equals(df2.iloc[:4])
        assert (df2.result.values[-2:] == np.array([880.0, 990.0])).all()


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
    rs = lambda n: "".join(letters[ii] for ii in ri(0, len(letters), n))

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

        ps.run_local(func, params, calc_dir=calc_dir, save=False)
        assert not os.path.exists(dbfn)
        assert os.listdir(tmpdir) == []

        ps.run_local(func, params, calc_dir=calc_dir, save=True)
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
        ps.run_local(func, params, calc_dir=calc_dir)

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
        return {"result": pset["a"] * 10}

    with tempfile.TemporaryDirectory() as tmpdir:
        calc_dir = pj(tmpdir, "calc")
        params = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]

        # First run. backup does nothing yet
        df0 = ps.run_local(func, params, calc_dir=calc_dir)
        unq = df0._run_id.unique()
        assert len(unq) == 1
        run_id_0 = unq[0]

        # Second run. This time, test backup.
        df1 = ps.run_local(func, params, calc_dir=calc_dir, backup=True)
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
        assert (dfa.result.values == dfb.result.values).all()
        assert (dfa._pset_seq.values == dfb._pset_seq.values).all()
        assert (dfa._run_seq.values == dfb._run_seq.values).all()
        assert (dfa._pset_hash.values == dfb._pset_hash.values).all()

    with tempfile.TemporaryDirectory() as tmpdir:
        calc_dir = pj(tmpdir, "calc")
        params = ps.plist("a", [1, 2, 3, 4])

        # no db disk write for now, test passing in no df, df=None and empty df
        df1_1 = ps.run_local(func, params, calc_dir=calc_dir, save=False)
        df1_2 = ps.run_local(
            func, params, calc_dir=calc_dir, save=False, df=None
        )
        df_cmp(df1_1, df1_2)
        df1_3 = ps.run_local(
            func, params, calc_dir=calc_dir, save=False, df=pd.DataFrame()
        )
        df_cmp(df1_1, df1_3)

        # still no disk write, pass in df1 and extend
        df1 = df1_3
        df2 = ps.run_local(func, params, calc_dir=calc_dir, save=False, df=df1)
        assert not os.path.exists(pj(tmpdir, "calc"))
        assert len(df2) == 2 * len(df1)
        assert (df2.a.values == np.tile(df1.a.values, 2)).all()
        assert (
            df2._pset_hash.values == np.tile(df1._pset_hash.values, 2)
        ).all()

        # df2 again, but now write to disk and read (ignore that run_local()
        # also returns it)
        ps.run_local(func, params, calc_dir=calc_dir, df=df1)
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


class _Foo:
    """Dummy custom type used in test_dotdict(). As always, must be defined
    outside b/c Python can't pickle stuff defined in the same scope.
    """

    x = 55

    def __eq__(self, other):
        return self.x == other.x

    def __hash__(self):
        return self.x


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

    f = _Foo()
    ref = dict(a=1, b=2, c=np.sin, d=f, e=_Foo)

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

    assert (
        ps.pset_hash(dict(a=np.sin))
        == "32522716b0514eb8b2677a917888c4ca21f792da"
    )

    # These are the correct hashes (tested: ipython or run this function in a
    # script). But pytest's deep introspection "magic" must do something to the
    # _Foo class such that its hash changes, thank you very much. So we just
    # test that hashing them works.
    #
    ##assert ps.pset_hash(dict(a=_Foo)) == "f7a936832c167d5dc1c9574b937822ad0853c00d"
    ##assert ps.pset_hash(dict(a=_Foo())) == "7c78a7c995e1cf7a2f1f68f29090b173c9b930fa"
    ps.pset_hash(dict(a=_Foo))
    ps.pset_hash(dict(a=_Foo()))

    d_no_us = dict(a=1, b=2)
    d_us = dict(a=1, b=2, _c=3)
    assert ps.pset_hash(d_no_us) == ps.pset_hash(d_us)
    assert ps.pset_hash(d_no_us) == ps.pset_hash(d_us, skip_special_cols=True)
    assert not ps.pset_hash(d_no_us) == ps.pset_hash(
        d_us, skip_special_cols=False
    )


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
    c = ps.plist("c", [np.sin, np.cos, _Foo, _Foo()])
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
    template_dir = f"{here}/../examples/batch_with_git/templates"
    with tempfile.TemporaryDirectory() as tmpdir:
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

        for pset_id in df._pset_id.values:
            for name in [
                "run.py",
                "pset.pk",
                "jobscript_local",
                "jobscript_cluster",
            ]:
                assert os.path.exists(f"{tmpdir}/calc/{pset_id}/{name}")
