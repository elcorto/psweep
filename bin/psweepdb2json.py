#!/usr/bin/env python3

import os

import docopt
import pandas as pd
from psweep import psweep as ps

__doc__ = r"""
Convert psweep databse pickle file to json. Use this for quick queries of the
databse, e.g. using jq in the shell.

usage:
    {this} [-o <orient>] <file>

options:
    -o <orient>  "DataFrame.to_json(orient=<orient>)" [default: records]

example:
    Nice formatting using jq
        $ {this} results.pk | jq . -C | less -R
    
    Find all _run_id
        $ {this} results.pk | jq -r '.[]|._run_id' | sort -u
        02ca9694-696e-4fdd-ac08-8c343080bb63
        0a1fb364-8681-4178-869e-1126f3719da4
        97058ee2-2e81-426f-b674-04b7ec718c43

    Complex example: show the start time of each run.
        $ {this} results.pk > /tmp/json
        $ for x in $(jq -r '.[]|._run_id' /tmp/json | sort -u); do \
        ... echo $x $(jq -r "[.[]|select(._run_id==\"$x\")|._time_utc]|min" /tmp/json)
        ... done
        02ca9694-696e-4fdd-ac08-8c343080bb63 2018-09-03T00:00:24Z
        0a1fb364-8681-4178-869e-1126f3719da4 2018-09-02T22:23:47Z
        97058ee2-2e81-426f-b674-04b7ec718c43 2018-09-02T22:09:44Z
""".format(this=os.path.basename(__file__))

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    df = ps.df_read(args['<file>'])
    print(ps.df_to_json(df, orient=args['-o']))
