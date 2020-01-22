#!/bin/bash
# setup environmental variables
EEGNET_WOLF_SRC_PATH="$(pwd)/../src"
PYTHONPATH="$PYTHONPATH:$(pwd)/../python_utils"

python3 run_test.py $1
