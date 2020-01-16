#! /bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

find . -type f -name "gen_test.py" | xargs python3
