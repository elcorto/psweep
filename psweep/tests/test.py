import subprocess as sp
import os, tempfile, io
from psweep import psweep as ps
import pandas as pd
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
    df = pd.DataFrame()
    params = [{'a': 1}, {'a': 2}]
    df1 = ps.run(df, func, params)
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
