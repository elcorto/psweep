import subprocess as sp
import os, tempfile
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
