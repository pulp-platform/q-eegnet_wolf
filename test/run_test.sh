#! /bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/../python_utils

find . -type f -name "gen_test.py" | xargs python3
