import io
import os
import re
import shutil
import string
import tempfile

import pandas as pd
import numpy as np

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
    dr = os.path.abspath("{}/../../../examples".format(here))
    for basename in os.listdir(dr):
        path = pj(dr, basename)
        print(f"running example: {path}")
        if basename.endswith(".py"):
            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = f"""
                    cp {path} {tmpdir}/; cd {tmpdir};
                    python3 {path}
                """
                print(system(cmd))
        elif os.path.isdir(path):
            with tempfile.TemporaryDirectory() as tmpdir:
                shutil.copytree(path, tmpdir, dirs_exist_ok=True)
                cmd = f"cd {tmpdir}; ./run_example.sh; ./clean.sh"
                print(system(cmd))


def test_shell_call():
    print(system("ls"))
    with tempfile.TemporaryDirectory() as tmpdir:
        txt = "ls"
        fn = pj(tmpdir, "test.sh")
        ps.file_write(fn, txt)
        cmd = f"cd {tmpdir}; sh test.sh"
        print(system(cmd))


def test_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        params = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
        calc_dir = "{}/calc".format(tmpdir)

        # run two times, updating the database, the second time,
        # also write tmp results
        df = ps.run_local(func, params, calc_dir=calc_dir)
        assert len(df) == 4
        assert len(df._run_id.unique()) == 1
        assert len(df._pset_id.unique()) == 4
        df = ps.run_local(
            func, params, calc_dir=calc_dir, poolsize=2, tmpsave=True
        )
        assert len(df) == 8
        assert len(df._run_id.unique()) == 2
        assert len(df._pset_id.unique()) == 8
        assert set(df.columns) == set(
            [
                "_calc_dir",
                "_pset_id",
                "_run_id",
                "_pset_seq",
                "_run_seq",
                "_pset_sha1",
                "_time_utc",
                "a",
                "result",
            ]
        )

        dbfn = "{}/database.pk".format(calc_dir)
        assert os.path.exists(dbfn)
        assert df.equals(ps.df_read(dbfn))

        # tmp results of second run
        run_id = df._run_id.unique()[-1]
        for pset_id in df[df._run_id == run_id]._pset_id:
            tmpsave_fn = "{calc_dir}/tmpsave/{run_id}/{pset_id}.pk".format(
                calc_dir=calc_dir, run_id=run_id, pset_id=pset_id
            )
            assert os.path.exists(tmpsave_fn)


def test_simulate():
    with tempfile.TemporaryDirectory() as tmpdir:
        params = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
        params_sim = [{"a": 88}, {"a": 99}]
        calc_dir = "{}/calc".format(tmpdir)
        calc_dir_sim = calc_dir + ".simulate"

        df = ps.run_local(func, params, calc_dir=calc_dir)
        df_sim = ps.run_local(
            func, params_sim, calc_dir=calc_dir, simulate=True
        )
        dbfn = "{}/database.pk".format(calc_dir)
        dbfn_sim = "{}/database.pk".format(calc_dir_sim)

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
                '"{}"'.format(rs(5)),
                "'{}'".format(rs(5)),
                (ri(0, 99), rn(), "{}".format(rs(5))),
                [ri(0, 99), rn(), "{}".format(rs(5))],
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
            df = df.append(row, ignore_index=True)

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
                        ps.df_write(df, fd.name, fmt=fmt, orient=orient)
                        read = ps.df_read(fd.name, fmt=fmt, orient=orient)
                    else:
                        ps.df_write(df, fd.name, fmt=fmt)
                        read = ps.df_read(fd.name, fmt=fmt)
                    assert_frame_equal(df, read, check_exact=False)
        elif fmt == "pickle":
            with tempfile.NamedTemporaryFile() as fd:
                ps.df_write(df, fd.name, fmt=fmt)
                read = ps.df_read(fd.name, fmt=fmt)
                assert_frame_equal(df, read)
        else:
            raise Exception("unknown fmt")


def test_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        params = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
        calc_dir = "{}/calc".format(tmpdir)
        dbfn = "{}/database.pk".format(calc_dir)

        df = ps.run_local(func, params, calc_dir=calc_dir, save=False)
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
        calc_dir = "{}/calc".format(tmpdir)
        ps.run_local(func, params, calc_dir=calc_dir)

        db = pj(calc_dir, "database.pk")
        print(system(f"psweep-db2json -o columns {db}"))
        print(
            system(f"psweep-db2table -i -a -f simple {db}")
        )


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
