import subprocess as sp
import os, tempfile, io, string, shutil, re

import pandas as pd
import numpy as np

from psweep import psweep as ps

pj = os.path.join
here = os.path.abspath(os.path.dirname(__file__))


def system(cmd):
    try:
        proc = sp.run(cmd, shell=True, check=True, stdout=sp.PIPE,
                      stderr=sp.STDOUT)
        out = proc.stdout.strip().decode()
    except sp.CalledProcessError as err:
        print(err.output.decode())
        raise
    return out


def test_run_all_examples():
    dr = os.path.abspath('{}/../../examples'.format(here))
    for basename in os.listdir(dr):
        if basename.endswith('.py'):
            tmpdir = tempfile.mkdtemp(prefix='psweep_test_examples_')
            cmd = """
                cp {fn} {tmpdir}/; cd {tmpdir};
                python3 {fn}
            """.format(tmpdir=tmpdir, fn=pj(dr, basename))
            print(system(cmd))
            shutil.rmtree(tmpdir)


def func(pset):
    # We need to multiply by a float here to make sure the 'result' column has
    # float dtype. Else the column will be cast to float once we add NaNs,
    # which breaks df.equals(other_df) .
    return {'result': pset['a']*10.0}


def test_run():
    tmpdir = tempfile.mkdtemp(prefix='psweep_test_run_')
    params = [{'a': 1}, {'a': 2}, {'a': 3}, {'a': 4}]
    calc_dir = "{}/calc".format(tmpdir)

    # run two times, updating the database, the second time,
    # also write tmp results
    df = ps.run(func, params, calc_dir=calc_dir)
    assert len(df) == 4
    assert len(df._run_id.unique()) == 1
    assert len(df._pset_id.unique()) == 4
    df = ps.run(func, params, calc_dir=calc_dir, poolsize=2, tmpsave=True)
    assert len(df) == 8
    assert len(df._run_id.unique()) == 2
    assert len(df._pset_id.unique()) == 8
    assert set(df.columns) == \
        set(['_calc_dir', '_pset_id', '_run_id', '_time_utc', 'a', 'result'])

    dbfn = "{}/results.pk".format(calc_dir)
    assert os.path.exists(dbfn)
    assert df.equals(ps.df_read(dbfn))

    # tmp results of second run
    run_id = df._run_id.unique()[-1]
    for pset_id in df[df._run_id==run_id]._pset_id:
        tmpsave_fn = "{calc_dir}/tmpsave/{run_id}/{pset_id}.pk".format(
            calc_dir=calc_dir, run_id=run_id, pset_id=pset_id)
        assert os.path.exists(tmpsave_fn)
    shutil.rmtree(tmpdir)


def test_is_seq():
    no = [{'a':1}, io.IOBase(), '123']
    yes = [[1,2], {1,2}, (1,2)]
    for obj in no:
        print(obj)
        assert not ps.is_seq(obj)
    for obj in yes:
        print(obj)
        assert ps.is_seq(obj)


def test_df_io():
    from pandas.util.testing import assert_frame_equal
    letters = string.ascii_letters
    ri = np.random.randint
    rn = np.random.rand
    # random string
    rs = lambda n: ''.join(letters[ii] for ii in ri(0, len(letters), n))

    for fmt in ['pickle', 'json']:
        df = pd.DataFrame()
        for _ in range(2):
            vals = [ri(0,100),
                    rs(5),
                    np.nan,
                    '"{}"'.format(rs(5)),
                    "'{}'".format(rs(5)),
                    (ri(0,99), rn(), '{}'.format(rs(5))),
                    [ri(0,99), rn(), "{}".format(rs(5))],
                    rn(),
                    rn(5),
                    rn(5,5),
                    list(rn(5)),
                    {'a':1, 'b':3, 'c': [1,2,3]},
                    ]
            if fmt == 'pickle':
                vals += [True,
                         False,
                         None,
                         set(ri(0,99,10)),
                         ]
            row = pd.DataFrame([dict(zip(letters, vals))])
            df = df.append(row, ignore_index=True)

        if fmt == 'json':
            for orient in [None, 'split', 'records', 'index', 'columns', '_default_']:
                print("orient: ", orient)
                fn = tempfile.mktemp(prefix='psweep_test_df_io_{}_{}_'.format(fmt, orient))
                if orient != '_default_':
                    ps.df_write(df, fn, fmt=fmt, orient=orient)
                    read = ps.df_read(fn, fmt=fmt, orient=orient)
                else:
                    ps.df_write(df, fn, fmt=fmt)
                    read = ps.df_read(fn, fmt=fmt)
                os.remove(fn)
                assert_frame_equal(df, read, check_exact=False, check_less_precise=12)
        elif fmt == 'pickle':
            fn = tempfile.mktemp(prefix='psweep_test_df_io_{}_'.format(fmt))
            ps.df_write(df, fn, fmt=fmt)
            read = ps.df_read(fn, fmt=fmt)
            os.remove(fn)
##            assert_frame_equal(df, read, check_exact=True)
            assert_frame_equal(df, read)
        else:
            raise Exception("unknown fmt")


def test_save():
    tmpdir = tempfile.mkdtemp(prefix='psweep_test_run_')
    params = [{'a': 1}, {'a': 2}, {'a': 3}, {'a': 4}]
    calc_dir = "{}/calc".format(tmpdir)
    dbfn = "{}/results.pk".format(calc_dir)

    df = ps.run(func, params, calc_dir=calc_dir, save=False)
    assert not os.path.exists(dbfn)
    assert os.listdir(tmpdir) == []
    
    df = ps.run(func, params, calc_dir=calc_dir, save=True)
    assert os.path.exists(dbfn)
    assert os.listdir(tmpdir) != []
    shutil.rmtree(tmpdir)


def test_merge_dicts():
    a = {'a': 1}
    b = {'b': 2}
    c = {'c': 3}

    # API
    m1 = ps.merge_dicts(a, b, c)
    m2 = ps.merge_dicts([a, b, c])

    # correct merge
    assert set(m1.keys()) == set(m2.keys())
    assert set(m1.values()) == set(m2.values())
    assert set(m1.keys()) == set(('a', 'b', 'c'))
    assert set(m1.values()) == set((1, 2, 3))

    # left-to-right
    a = {'a': 1}
    b = {'a': 2}
    c = {'a': 3}
    m = ps.merge_dicts(a, b)
    assert m == {'a': 2}
    m = ps.merge_dicts(a, b, c)
    assert m == {'a': 3}


def test_backup():
    def func(pset):
        # write stuff to calc_dir
        dr = pj(pset['_calc_dir'], pset['_pset_id'])
        ps.makedirs(dr)
        fn = pj(dr, 'foo')
        with open(fn, 'w') as fd:
            fd.write("bar")
        return ps.merge_dicts(pset, {'result': pset['a']*10})

    tmpdir = tempfile.mkdtemp(prefix='psweep_test_run_')
    params = [{'a': 1}, {'a': 2}, {'a': 3}, {'a': 4}]
    calc_dir = "{}/calc".format(tmpdir)
    
    # First run. backup_calc_dir does nothing yet. Test backup_script.
    df = ps.run(func, params, calc_dir=calc_dir, backup_script=__file__,
                backup_calc_dir=True)
    run_id = df._run_id.unique()[-1]
    script_fn = "{}/backup_script/{}.py".format(calc_dir, run_id)
    assert os.path.exists(script_fn)
    with open(script_fn) as fd1, open(__file__) as fd2:
        assert fd1.read() == fd2.read()
    df1 = df.copy()  

    # Second run. This time, test backup_calc_dir.
    df = ps.run(func, params, calc_dir=calc_dir, backup_calc_dir=True)
    rex = re.compile(r"calc_[0-9-]+T[0-9:\.]+Z")
    found = False
    for name in os.listdir(tmpdir):
        if rex.search(name) is not None:
            backup_dir = pj(tmpdir, name)
            found = True
            break
    assert found
    print(os.listdir(backup_dir))
    for pset_id in df1[df1._run_id==run_id]._pset_id:
        tgt = pj(backup_dir, pset_id, 'foo')
        assert os.path.exists(tgt)
    assert os.path.exists(pj(backup_dir, 'results.pk'))
    assert os.path.exists("{}/backup_script/{}.py".format(backup_dir, run_id))
