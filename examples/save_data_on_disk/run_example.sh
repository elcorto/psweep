#!/bin/sh

set -eu

./clean.sh

./10run.py
./20eval.py
