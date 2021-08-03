#!/bin/sh

set -eux

./clean.sh

./10run.py
./20run.py
