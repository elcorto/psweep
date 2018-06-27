import subprocess as sp
import os, tempfile, io, string

import pandas as pd
import numpy as np

from psweep import psweep as ps

pj = os.path.join


here = os.path.abspath(os.path.dirname(__file__))

def test_json2table():
    with open('{}/files/results.json.json2table'.format(here)) as fd:
        ref = fd.read().strip()
    exe = os.path.abspath('{}/../../bin/json2table.py'.format(here))
    cmd = '{exe} {here}/files/results.json'.format(exe=exe, here=here)
    val = sp.getoutput(cmd).strip()
    assert val == ref


def test_examples():
    dr = os.path.abspath('{}/../../examples'.format(here))
    tmpdir = tempfile.mkdtemp(prefix='psweep_test_examples')
    for basename in os.listdir(dr):
        cmd = """
            cp {fn} {tmpdir}/; cd {tmpdir}; 
            python3 {fn}
        """.format(tmpdir=tmpdir, fn=pj(dr, basename))
        print(sp.getoutput(cmd))


def test_run():
    def func(pset):
        return {'result': pset['a']*10}
    tmpdir = tempfile.mkdtemp(prefix='psweep_test_run')
    df = pd.DataFrame()
    params = [{'a': 1}, {'a': 2}]
    df1 = ps.run(df, func, params, tmpsave=pj(tmpdir, 'test_run_1'))
    for ii in range(len(params)):
        assert os.path.exists(pj(tmpdir, 'test_run_1.0.{}'.format(ii)))
    columns = ['_run', 'a', 'result']
    ref1 = pd.DataFrame([[0,1,10], 
                         [0,2,20]], 
                         columns=columns)
    assert len(df1.columns) == len(ref1.columns)
    assert ref1.equals(df1.reindex(columns=ref1.columns))
    params = [{'a': 3}, {'a': 4}]
    df2 = ps.run(df1, func, params)
    ref = pd.DataFrame([[1,3,30], 
                        [1,4,40]], 
                         columns=columns)
    ref2 = ref1.append(ref, ignore_index=True)
    assert len(df2.columns) == len(ref2.columns)
    assert ref2.equals(df2.reindex(columns=ref2.columns))
    assert (df2.index == pd.Int64Index([0,1,2,3], dtype='int64')).all()


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
##                'NaN', 
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
        row = [dict(zip(let, vals))]
        df = df.append(row, ignore_index=True)

    for orient in [None, 'split', 'records', 'index', 'columns', '_default_']:
        print("orient: ", orient)
        if orient != '_default_':
            json = df.to_json(orient=orient, double_precision=15)
            read = pd.io.json.read_json(json, orient=orient, precise_float=True)
            assert_frame_equal(df, read, check_exact=False, check_less_precise=12)
        try: 
            fn = tempfile.mktemp(prefix='psweep_test_df_json_io_{}_'.format(orient))
            if orient != '_default_':
                ps.df_json_write(df, fn, orient=orient)
                read = ps.df_json_read(fn, orient=orient)
            else:
                ps.df_json_write(df, fn)
                read = ps.df_json_read(fn)
            assert_frame_equal(df, read, check_exact=False, check_less_precise=12)
        finally:
            os.remove(fn)
