import subprocess as sp
import os, tempfile
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
    assert ref1.equals(df1)
    params = [{'a': 3}, {'a': 4}]
    df2 = ps.run(df1, func, params)
    ref = pd.DataFrame([[1,3,30], 
                        [1,4,40]], 
                         columns=columns)
    ref2 = ref1.append(ref, ignore_index=True)
    print(ref2)
    print(df2)
    assert ref2.equals(df2)
    assert (df2.index == pd.Int64Index([0,1,2,3], dtype='int64')).all()
