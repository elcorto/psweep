#!/usr/bin/env python3

# Pretty-print a json file written by 
#   DataFrame.to_json('results.json', orient='split')

import sys
from pandas.io.json import read_json

print(read_json(sys.argv[1], orient='split'))
