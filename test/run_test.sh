#!/bin/bash

PLATFORM="gvsoc"

while getopts "bp:h" name; do
    case "$name" in
        b) PLATFORM="board";;
        p) PLATFORM=$OPTARG;;
        h) printf "Usage: %s [-b] [-p platform] [root_folder]\n" $0
           printf " -b            build on the board, equivalent to -p board\n"
           printf " -p <platform> build on the desired platform [board | gvsoc], default is gvsoc\n"
           printf " -h            show this help message\n"
           printf " root_folder   Start folder where to execute all the tests\n"
           exit 0;;
        ?) printf "Usage: %s [-b] [-p platform] root_folder\n" $0
           exit 2;;
    esac
done

printf "Testing on Platform: %s\n\n" $PLATFORM

ROOT=${@:$OPTIND:1}


# setup environmental variables
PYTHONPATH="$PYTHONPATH:$(pwd)/../python_utils"

# set the platform
PULP_CURRENT_CONFIG_ARGS="platform=$PLATFORM"

# always store the trace file
# PULP_CURRENT_CONFIG_ARGS+=" gvsoc/trace=l2_priv:$(pwd)/../build/trace.txt"

python3 run_test.py $ROOT
