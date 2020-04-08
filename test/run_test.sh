#!/bin/bash
#
# Copyright (C) 2020 ETH Zurich. All rights reserved.
#
# Author: Tibor Schneider, ETH Zurich
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
