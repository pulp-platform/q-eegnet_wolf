#! /bin/bash

PLATFORM="gvsoc"
RUN=true
GTKWAVE=false

while getopts "bp:nwh" name; do
    case "$name" in
        b) PLATFORM="board";;
        p) PLATFORM=$OPTARG;;
        n) RUN=false;;
        w) GTKWAVE=true;;
        h) printf "Usage: %s [-b] [-p platform] [-h] [-n] [-w]\n" $0
           printf " -b            build on the board, equivalent to -p board\n"
           printf " -p <platform> build on the desired platform [board | gvsoc], default is gvsoc\n"
           printf " -n            do not run the program, just build it\n"
           printf " -w            generate GTK wave files\n"
           printf " -h            show this help message\n"
           exit 0;;
        ?) printf "Usage: %s [-b] [-p platform] root_folder\n" $0
           exit 2;;
    esac
done

printf "Running EEGnet on Platform: %s\n\n" $PLATFORM

# add python_utils to the python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/python_utils

# set the platform
PULP_CURRENT_CONFIG_ARGS="platform=$PLATFORM"

# enter data directory
cd data

# generate net header file
python3 gen_net_header.py
python3 gen_input_header.py

# leave data directory
cd ..

# build everything
make clean all

# run if requested
if [ "$GTKWAVE" = true ] ; then
    make run runner_args="--vcd --event=.*"
else
    if [ "$RUN" = true ] ; then
        make run
    fi
fi
