#! /bin/bash

# add python_utils to the python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/python_utils

# enter data directory
cd data

# generate net header file
python3 gen_net_header.py
python3 gen_input_header.py

# leave data directory
cd ..

# build and run everything
make clean all run
