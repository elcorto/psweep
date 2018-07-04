import subprocess as sp
import os, tempfile, io, string, json, shutil

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
    return {'result': pset['a']*10}


def test_run():
    tmpdir = tempfile.mkdtemp(prefix='psweep_test_run_')
    params = [{'a': 1}, {'a': 2}, {'a': 3}, {'a': 4}]
    calc_dir = f"{tmpdir}/calc"

    # run two times, updating the database .../results.json, the second time,
    # also write tmp results
    df = ps.run(func, params, calc_dir=calc_dir)
    assert len(df) == 4
    assert len(df._run_id.unique()) == 1
    assert len(df._pset_id.unique()) == 4
    df = ps.run(func, params, calc_dir=calc_dir, poolsize=2, tmpsave=True)
    assert len(df) == 8
    assert len(df._run_id.unique()) == 2
    assert len(df._pset_id.unique()) == 8

    dbfn = f"{calc_dir}/results.json"
    assert os.path.exists(dbfn)
    assert df.equals(ps.df_json_read(dbfn))
    assert set(df.columns) == \
        set(['_calc_dir', '_pset_id', '_run_id', 'a', 'result'])

    # tmp results of second run
    run_id = df._run_id.unique()[-1]
    for pset_id in df[df._run_id==run_id]._pset_id:
        tmpsave_fn = f"{calc_dir}/tmpsave/{run_id}/{pset_id}.json"
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


def test_df_json_io():
    from pandas.util.testing import assert_frame_equal
    let = string.ascii_letters
    ri = np.random.randint
    rn = np.random.rand
    rs = lambda n: ''.join(let[ii] for ii in ri(0, len(let), n))
    df = pd.DataFrame()
    for _ in range(2):
        vals = [ri(0,100),
                rs(5),
                np.nan,
                True,
                False,
                None,
                '"{}"'.format(rs(5)),
                "'{}'".format(rs(5)),
                (ri(0,99), rn(), '{}'.format(rs(5))),
                [ri(0,99), rn(), "{}".format(rs(5))],
##                set(ri(0,99,10)),
                rn(),
                rn(5),
                rn(5,5),
                list(rn(5)),
                {'a':1, 'b':3, 'c': [1,2,3]},
                ]
        row = pd.DataFrame([dict(zip(let, vals))], dtype=object)
        df = df.append(row, ignore_index=True)

    for orient in [None, 'split', 'records', 'index', 'columns', '_default_']:
        print("orient: ", orient)
        fn = tempfile.mktemp(prefix='psweep_test_df_json_io_{}_'.format(orient))
        if orient != '_default_':
            ps.df_json_write(df, fn, orient=orient)
            read = ps.df_json_read(fn, orient=orient)
        else:
            ps.df_json_write(df, fn)
            read = ps.df_json_read(fn)
        os.remove(fn)
        assert_frame_equal(df, read, check_exact=False, check_less_precise=12)
