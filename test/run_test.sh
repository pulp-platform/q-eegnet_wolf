#!/bin/bash
# setup environmental variables
EEGNET_WOLF_SRC_PATH="$(pwd)/../src"
PYTHONPATH="$PYTHONPATH:$(pwd)/../python_utils"

# always store the trace file
PULP_CURRENT_CONFIG_ARGS="platform=gvsoc gvsoc/trace=insn:$(pwd)/../build/trace.txt"

python3 run_test.py $1
