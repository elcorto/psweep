import argparse
import json
import inspect

from . import psweep as ps


def check_calc_dir():
    func = ps.check_calc_dir
    desc = f"""
{inspect.getdoc(func)}

Uses {func.__name__}() in {inspect.getsourcefile(func)}.
"""
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("calc_dir", metavar="CALC_DIR")
    args = parser.parse_args()
    calc_dir = args.calc_dir

    out = func(calc_dir, ps.df_read(f"{calc_dir}/database.pk"))
    _filt = lambda x: list(x) if isinstance(x, set) else x
    print(json.dumps(dict((kk, _filt(vv)) for kk, vv in out.items())))
